from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

log = logging.getLogger("FastH3VSA")

_RUNTIME_KEY = "fasth3_vsa_runtime_v2"
_PATCH_MARKER = "fasth3_vsa_patch_v2"
_WRAPPER_KEY = "fasth3_vsa_layout_v2"
_TILE = 64
_TILE_SHAPE = (4, 4, 4)


@dataclass
class _Geometry:
    untile_index: torch.Tensor       # original packed row -> padded tiled row
    block_len: torch.Tensor          # live rows in every 64-token tile
    num_prefix_tiles: int
    total_rows: int
    padded_rows: int


class _VSAState:
    def __init__(self, sparsity: float):
        self.sparsity = float(sparsity)
        self._geometry: Dict[Tuple, _Geometry] = {}
        self.calls = 0

    def geometry(self, prefix_segments, video_grid, device) -> _Geometry:
        key = (tuple(prefix_segments), tuple(video_grid), str(device))
        hit = self._geometry.get(key)
        if hit is not None:
            return hit

        prefix_segments = tuple(int(x) for x in prefix_segments if int(x) > 0)
        vt, vh, vw = (int(x) for x in video_grid)
        if min(vt, vh, vw) <= 0:
            raise RuntimeError(f"FastH3 VSA: invalid video token grid {video_grid}")

        prefix_len = sum(prefix_segments)
        video_len = vt * vh * vw
        total_rows = prefix_len + video_len

        # `untile[pos_original] = pos_padded_tile`.  Prefix tiles are segment-pure;
        # video tiles are the FastVideo tile-64 geometry (4,4,4).
        untile = torch.empty(total_rows, dtype=torch.long, device=device)
        live = []
        tile_id = 0
        src = 0

        # Prefix: do not let a 64-token tile straddle text/audio/condition segments.
        for seg_len in prefix_segments:
            left = seg_len
            while left:
                n = min(_TILE, left)
                dst0 = tile_id * _TILE
                untile[src:src+n] = torch.arange(dst0, dst0+n, device=device)
                live.append(n)
                src += n
                left -= n
                tile_id += 1
        num_prefix_tiles = tile_id

        # Generated video is flattened t-major, then h, then w in ComfyUI H3.
        ts_t, ts_h, ts_w = _TILE_SHAPE
        video_start = prefix_len
        for t0 in range(0, vt, ts_t):
            for h0 in range(0, vh, ts_h):
                for w0 in range(0, vw, ts_w):
                    original = []
                    for dt in range(ts_t):
                        t = t0 + dt
                        if t >= vt:
                            continue
                        for dh in range(ts_h):
                            h = h0 + dh
                            if h >= vh:
                                continue
                            for dw in range(ts_w):
                                w = w0 + dw
                                if w >= vw:
                                    continue
                                original.append(video_start + (t * vh + h) * vw + w)
                    n = len(original)
                    if not n:
                        continue
                    dst0 = tile_id * _TILE
                    idx = torch.tensor(original, dtype=torch.long, device=device)
                    untile[idx] = torch.arange(dst0, dst0+n, device=device)
                    live.append(n)
                    tile_id += 1

        if src != prefix_len:
            raise RuntimeError(f"FastH3 VSA: prefix accounting mismatch {src} != {prefix_len}")
        if len(live) != tile_id:
            raise RuntimeError("FastH3 VSA: internal tile accounting mismatch")
        block_len = torch.tensor(live, dtype=torch.int32, device=device)
        if int(block_len.sum()) != total_rows:
            raise RuntimeError(
                f"FastH3 VSA: tiled live rows sum to {int(block_len.sum())}, expected {total_rows}"
            )
        if int(torch.unique(untile).numel()) != total_rows:
            raise RuntimeError("FastH3 VSA: untile map is not injective")

        geo = _Geometry(
            untile_index=untile,
            block_len=block_len,
            num_prefix_tiles=num_prefix_tiles,
            total_rows=total_rows,
            padded_rows=tile_id * _TILE,
        )
        self._geometry[key] = geo
        log.info(
            "FastH3 VSA geometry: prefix=%s grid=%s tiles=%d (%d prefix), rows=%d -> %d padded",
            prefix_segments, video_grid, tile_id, num_prefix_tiles, total_rows, geo.padded_rows,
        )
        return geo


def _tile(x: torch.Tensor, geo: _Geometry) -> torch.Tensor:
    # x: [B,S,H,D]; padded slots remain zero. FastVideo uses the same transport contract.
    if x.ndim != 4 or x.shape[1] != geo.total_rows:
        raise RuntimeError(
            f"FastH3 VSA: expected [B,{geo.total_rows},H,D], got {tuple(x.shape)}"
        )
    out = x.new_zeros((x.shape[0], geo.padded_rows, x.shape[2], x.shape[3]))
    out[:, geo.untile_index] = x
    return out


def _executor_call(executor, *args, **kwargs):
    # IMPORTANT: wrappers receive a WrapperExecutor positioned at the current
    # wrapper. Calling .execute() re-enters the same wrapper forever. Calling
    # the executor object invokes WrapperExecutor.__call__(), which advances to
    # the next wrapper (or the original diffusion-model function).
    return executor(*args, **kwargs)


def _make_diffusion_wrapper(state: _VSAState, diffusion_model):
    def wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        from comfy.ldm.minimax.model import PackedLayout

        if transformer_options is None:
            transformer_options = {}
        payload = kwargs.get("minimax_payload") or {}
        video_x, audio_x = x[0], x[1]
        if video_x.ndim != 5 or int(video_x.shape[0]) != 1:
            raise RuntimeError("FastH3 VSA requires native batch-1 MiniMax H3 AV latents")

        pt, ph, pw = tuple(int(v) for v in diffusion_model.patch_size)
        latent_t = int(math.ceil(video_x.shape[2] / pt) * pt)
        latent_h = int(math.ceil(video_x.shape[3] / ph) * ph)
        latent_w = int(math.ceil(video_x.shape[4] / pw) * pw)
        audio_t = int(audio_x.shape[-1])
        text_len = int(context.shape[1])

        layout = payload.get("layout")
        signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        if layout is None or tuple(getattr(layout, "signature", ())) != signature:
            layout = PackedLayout(
                text_len, latent_t, latent_h, latent_w, audio_t,
                keyframes=payload.get("keyframes"), refs=payload.get("refs"),
            )

        segments = list(layout.segments)
        video_segments = [s for s in segments if s[2] == "video"]
        if len(video_segments) != 1:
            raise RuntimeError(f"FastH3 VSA expected one target video segment, got {video_segments}")
        va, vb, _ = video_segments[0]
        if vb != int(layout.seq_len):
            raise RuntimeError("FastH3 VSA expects generated video to be the final packed segment")

        prefix_segments = tuple(int(b-a) for a, b, kind in segments if kind != "video")
        grid = (latent_t // pt, latent_h // ph, latent_w // pw)
        expected_video = grid[0] * grid[1] * grid[2]
        if int(vb-va) != expected_video:
            raise RuntimeError(
                f"FastH3 VSA video rows {vb-va} do not match token grid {grid} ({expected_video})"
            )

        runtime = {
            "prefix_segments": prefix_segments,
            "video_grid": grid,
            "layout_seq_len": int(layout.seq_len),
        }
        prior = transformer_options.get(_RUNTIME_KEY)
        transformer_options[_RUNTIME_KEY] = runtime
        try:
            return _executor_call(executor, x, timestep, context, transformer_options, **kwargs)
        finally:
            if prior is None:
                transformer_options.pop(_RUNTIME_KEY, None)
            else:
                transformer_options[_RUNTIME_KEY] = prior

    return wrapper


def _vsa_attention(attn, x, rope_freqs, transformer_options, state: _VSAState):
    import comfy.model_management
    import comfy.quant_ops
    import comfy_kitchen as ck

    runtime = transformer_options.get(_RUNTIME_KEY)
    if runtime is None:
        raise RuntimeError("FastH3 VSA runtime metadata is missing; diffusion wrapper did not run")
    if not hasattr(attn, "to_gate_compress"):
        raise RuntimeError(
            "FastH3 VSA checkpoint gate is missing (attn.to_gate_compress). "
            "Apply ComfyUI PR #15958 and load the FastVideo VSA checkpoint."
        )

    s = int(x.shape[0])
    inner = int(attn.heads * attn.head_dim)
    geo = state.geometry(runtime["prefix_segments"], runtime["video_grid"], x.device)
    if geo.total_rows != s:
        raise RuntimeError(f"FastH3 VSA geometry has {geo.total_rows} rows but attention got {s}")

    # Tile BEFORE token-wise projections. qkv_proj and to_gate_compress have no bias,
    # so zero pad rows stay zero. This is algebraically equivalent to projecting in
    # native packed order and scattering q/k/v/gate afterwards, while avoiding four
    # additional sequence-sized copies on 24 GB consumer GPUs.
    x_t = x.new_zeros((geo.padded_rows, x.shape[-1]))
    x_t[geo.untile_index] = x
    sp = geo.padded_rows

    q, k, v = attn.qkv_proj(x_t).split(inner, dim=-1)
    v = v.view(1, sp, attn.heads, attn.head_dim).clone()

    if rope_freqs is not None:
        # RoPE is token-local too. Pad rotation entries can be zero because the
        # corresponding q/k pad rows are exactly zero and are never live keys.
        rope_t = rope_freqs.new_zeros((rope_freqs.shape[0], sp, *rope_freqs.shape[2:]))
        rope_t[:, geo.untile_index] = rope_freqs
        q = q.view(1, sp, attn.heads, attn.head_dim)
        k = k.view(1, sp, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_t.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_t, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
        )
    else:
        q = attn.q_norm(q.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)
        k = attn.k_norm(k.view(sp, attn.heads, attn.head_dim)).unsqueeze(0)

    gate_t = attn.to_gate_compress(x_t).view(1, sp, attn.heads, attn.head_dim)

    # The current comfy-kitchen CUDA Sol/VSA kernel is BF16/head_dim=128. Fail loudly
    # instead of silently selecting the O(S^2) eager reference implementation.
    if q.device.type != "cuda" or q.dtype != torch.bfloat16 or int(attn.head_dim) != 128:
        raise RuntimeError(
            "FastH3 VSA CUDA path requires CUDA BF16 with head_dim=128; "
            f"got device={q.device}, dtype={q.dtype}, head_dim={attn.head_dim}"
        )

    # FastVideo VSA-H3 'exempt' contract:
    # - prefix/non-video query tiles are dense        -> sink_q
    # - prefix/non-video key tiles are always kept   -> sink_blocks
    # - 90% sparsity over VIDEO key tiles            -> topk_ratio=0.10
    # - no pooled Sol tail; use the trained coarse gate branch instead.
    keep_ratio = max(0.0, min(1.0, 1.0 - state.sparsity))
    with ck.use_backend("cuda"):
        out_t = ck.sol_attn(
            q, k, v,
            scale=attn.head_dim ** -0.5,
            sink_blocks=[0, geo.num_prefix_tiles],
            sink_q=[0, geo.num_prefix_tiles],
            topk_ratio=keep_ratio,
            tail=False,
            block_len=geo.block_len,
            coarse_gate=gate_t,
        )

    out = out_t[:, geo.untile_index]  # padded tile order -> native packed order
    out = out.reshape(s, inner)
    state.calls += 1
    return attn.out_proj(out)

def _make_block_replace(block, state: _VSAState):
    def replace(args, extra):
        # Reproduce ComfyUI MiniMax H3 DiTBlock.forward exactly, replacing only
        # the attention call with tile-64 VSA + trained coarse gate.
        from comfy.ldm.minimax import model as mm

        x = args["img"]
        t_emb = args["t_emb"]
        mod_segments = args["mod_segments"]
        rope_freqs = args["rope_freqs"]
        transformer_options = args["transformer_options"]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = mm._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
        attn_out = _vsa_attention(block.attn, h, rope_freqs, transformer_options, state)
        x = mm._mod_gate(x, gate_msa, attn_out, mod_segments)
        h = mm._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
        x = mm._mod_gate(x, gate_mlp, block.mlp(h), mod_segments)
        return {"img": x}

    return replace


class FastH3VSAPatch:
    """FastVideo FastH3 VSA-H3 semantics on ComfyUI using comfy-kitchen Sol/VSA CUDA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sparsity": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 0.98, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "FastH3/VSA"
    DESCRIPTION = (
        "FastH3 VSA-H3 tile-64 patch for MiniMax H3. Uses segment-pure prefix tiles, "
        "4x4x4 video tiles, prefix-dense/exempt routing, and the checkpoint's trained "
        "to_gate_compress coarse branch through comfy-kitchen."
    )

    def patch(self, model, sparsity=0.90):
        sparsity = float(sparsity)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0,1)")

        patched = model.clone()
        options = patched.model_options
        if options.get(_PATCH_MARKER):
            raise RuntimeError("FastH3 VSA patch is already installed on this MODEL")

        transformer = options.setdefault("transformer_options", {})
        if transformer.get("optimized_attention_override") is not None:
            raise RuntimeError(
                "FastH3 VSA cannot stack with another optimized_attention override. "
                "Remove SageAttention/Sol-Attn/other attention patch nodes from this model branch."
            )
        existing = (transformer.get("patches_replace", {}).get("dit", {}))
        if existing:
            raise RuntimeError(
                "FastH3 VSA requires an unmodified H3 DiT block chain; another dit patch_replace is already installed."
            )

        dm = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if dm is None or type(dm).__name__ != "MiniMaxH3Model":
            raise ValueError("FastH3 VSA requires a native ComfyUI MiniMaxH3Model")
        blocks = list(getattr(dm, "blocks", ()))
        if len(blocks) != 50:
            raise RuntimeError(f"FastH3 VSA expected 50 H3 DiT blocks, found {len(blocks)}")
        object_patches = set(getattr(patched, "object_patches", {}))
        missing = [
            i for i, b in enumerate(blocks)
            if not hasattr(b.attn, "to_gate_compress")
            and f"diffusion_model.blocks.{i}.attn.to_gate_compress" not in object_patches
        ]
        if missing:
            raise RuntimeError(
                "FastH3 VSA compression gate is absent from blocks " + str(missing[:8]) +
                ". Apply ComfyUI PR #15958 and use the FastVideo VSA checkpoint, "
                "or attach transplanted gate object patches first."
            )

        try:
            import comfy_kitchen as ck
            backends = ck.list_backends()
            if not backends.get("cuda", {}).get("available", False):
                raise RuntimeError("comfy-kitchen CUDA backend is unavailable")
            if "sol_attn" not in backends.get("cuda", {}).get("capabilities", []):
                raise RuntimeError("comfy-kitchen CUDA backend does not expose sol_attn")
        except Exception as exc:
            raise RuntimeError(f"FastH3 VSA requires the comfy-kitchen Sol-Attn CUDA build: {exc}") from exc

        state = _VSAState(sparsity=sparsity)
        patched.add_wrapper_with_key(
            __import__("comfy.patcher_extension", fromlist=["WrappersMP"]).WrappersMP.DIFFUSION_MODEL,
            _WRAPPER_KEY,
            _make_diffusion_wrapper(state, dm),
        )
        for i, block in enumerate(blocks):
            patched.set_model_patch_replace(_make_block_replace(block, state), "dit", "double_block", i)

        options[_PATCH_MARKER] = {"version": 2, "sparsity": sparsity, "tile_size": 64}
        log.info(
            "FastH3 VSA installed: 50 blocks, sparsity=%.2f, tile=64 (4x4x4), coarse gate=trained",
            sparsity,
        )
        return (patched,)


def _extract_gate_weights(state):
    """Return {block_index: 2D gate_weight} from known FastH3 gate namespaces."""
    import re

    patterns = (
        re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.set_weight$"),
        re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
        re.compile(r"^diffusion_model\.blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
        re.compile(r"^blocks\.(\d+)\.attn\.to_gate_compress\.weight$"),
    )

    gates = {}
    for key, value in state.items():
        block_index = None
        for pattern in patterns:
            match = pattern.match(str(key))
            if match:
                block_index = int(match.group(1))
                break
        if block_index is None:
            continue
        if value.ndim != 2:
            raise ValueError(
                f"FastH3 VSA gate must be a matrix, got {key} with shape {tuple(value.shape)}"
            )
        if block_index in gates:
            raise ValueError(f"duplicate FastH3 VSA gate for block {block_index}")
        gates[block_index] = value
    return gates


class Ref2VAVSAGatePatch:
    """
    Experimental Ref2VA VSA transplant using an external FastH3 gate file.

    The existing FastH3 layout code already protects every non-target-video
    segment as a dense/exempt prefix. Ref2VA references therefore remain dense;
    VSA top-k is applied only over generated-video key tiles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL",),
                "gate_file": (folder_paths.get_filename_list("loras"),),
                "sparsity": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.98, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "FastH3/VSA"
    DESCRIPTION = (
        "EXPERIMENTAL Ref2VA VSA. Loads 50 trained FastH3 to_gate_compress matrices "
        "from a gate safetensors file, attaches them to the Ref2VA H3 blocks, then "
        "runs the same tile-64 Sol/VSA kernel. Text, reference images/video/audio "
        "and target audio stay dense/exempt; only generated-video key tiles are sparse. "
        "Start at sparsity 0.50-0.70 and compare against the same dense seed."
    )

    def patch(self, model, gate_file, sparsity=0.70):
        import folder_paths
        import comfy.utils
        from torch import nn

        sparsity = float(sparsity)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0,1)")

        gate_path = folder_paths.get_full_path_or_raise("loras", gate_file)
        raw = comfy.utils.load_torch_file(gate_path, safe_load=True)
        gates = _extract_gate_weights(raw)

        patched = model.clone()
        dm = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if dm is None or type(dm).__name__ != "MiniMaxH3Model":
            raise ValueError("Ref2VA VSA requires a native ComfyUI MiniMaxH3Model")

        blocks = list(getattr(dm, "blocks", ()))
        if len(blocks) != 50:
            raise RuntimeError(f"Ref2VA VSA expected 50 H3 DiT blocks, found {len(blocks)}")

        expected = set(range(len(blocks)))
        actual = set(gates)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "Gate file must contain exactly one to_gate_compress matrix for every H3 block; "
                f"gates={len(gates)}, missing={missing[:8]}, extra={extra[:8]}"
            )

        existing_patches = set(getattr(patched, "object_patches", {}))
        shapes = set()

        for index, block in enumerate(blocks):
            weight = gates[index]
            out_features, in_features = map(int, weight.shape)

            hidden = int(getattr(block.attn.qkv_proj, "in_features", in_features))
            inner = int(getattr(block.attn, "heads", 0)) * int(
                getattr(block.attn, "head_dim", 0)
            )
            if in_features != hidden or out_features != inner:
                raise ValueError(
                    "FastH3 VSA gate shape does not match this Ref2VA base: "
                    f"block={index}, gate={tuple(weight.shape)}, "
                    f"expected=({inner}, {hidden})"
                )

            path = f"diffusion_model.blocks.{index}.attn.to_gate_compress"
            if path in existing_patches:
                raise ValueError(f"a model object patch already owns {path}")

            gate = nn.Linear(
                in_features,
                out_features,
                bias=False,
                device=weight.device,
                dtype=weight.dtype,
            )
            gate.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
            gate.eval()
            setattr(block.attn, "to_gate_compress", gate)
            shapes.add((out_features, in_features))

        # `FastH3VSAPatch.patch()` clones model options, preserving object patches.
        # Its gate validation accepts the pending object patches added above; they
        # are materialized by Comfy before diffusion execution.
        (patched,) = FastH3VSAPatch().patch(patched, sparsity=sparsity)
        patched.model_options[_PATCH_MARKER]["gate_source"] = str(gate_file)
        patched.model_options[_PATCH_MARKER]["experimental_ref2va"] = True
        patched.model_options[_PATCH_MARKER]["gate_shapes"] = [list(s) for s in sorted(shapes)]

        log.warning(
            "EXPERIMENTAL Ref2VA VSA installed: gate_file=%s, 50 gates, sparsity=%.2f. "
            "Refs/text/audio are dense prefix; generated video uses VSA top-k.",
            gate_file,
            sparsity,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "FastH3VSAPatch": FastH3VSAPatch,
    "Ref2VAVSAGatePatch": Ref2VAVSAGatePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastH3VSAPatch": "FastH3 VSA-H3 Patch (tile64)",
    "Ref2VAVSAGatePatch": "Ref2VA VSA Gate Transplant (EXPERIMENTAL)",
}
