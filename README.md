# ComfyUI-Ref2VA-VSA: Ultra-Fast Character Video Generation (72s on RTX 4090)

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom%20Node-blue?style=for-the-badge" alt="ComfyUI">
  <img src="https://img.shields.io/badge/Model-MiniMax--H3%20Ref2VA-orange?style=for-the-badge" alt="MiniMax-H3">
  <img src="https://img.shields.io/badge/Speed-72s%20%2F%205s%20Video-brightgreen?style=for-the-badge" alt="Speed">
  <img src="https://img.shields.io/badge/Hardware-RTX%204090%20(24GB)-purple?style=for-the-badge" alt="Hardware">
</p>

> **Breakthrough inference speed for MiniMax H3 Reference-to-Video (Ref2VA / R2VA)**: Generate a full 5-second, 24 fps video conditioned on reference character images in **~72s (warm) / 95s (first run)** on a single consumer **NVIDIA GeForce RTX 4090 (24GB)**.

---

## 🎬 Empirical Benchmark & Results

Both videos were generated on the exact same **NVIDIA GeForce RTX 4090 (24GB)**, using the exact same prompt, character reference image, seed (`981445682258077`), and resolution (**1344×768, 5.16s @ 24 fps / 124 frames**).

<p align="center">
  <img src="assets/example_character.jpg" width="220" style="border-radius: 8px;" alt="Input Reference Character">
  <br>
  <b>Input Character Reference Image</b>
</p>

<table>
  <thead>
    <tr align="center">
      <th width="50%">
        <h3>🚀 Ref2VA VSA (Ours - 4 Steps)</h3>
        <b>⏱️ 95s (1st run) / ~72s (warm)</b>
      </th>
      <th width="50%">
        <h3>🔬 Video Delta Net / VDN-H3 (8 Steps)</h3>
        <b>⏱️ 213s (1st run) / ~135s (warm)</b>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img src="assets/vsa_ref2va_preview.webp" width="100%" alt="Ref2VA VSA Preview">
        <br>
        <a href="assets/vsa_ref2va_4step_95s.mp4"><b>▶ Download Full 1344x768 Video (MP4)</b></a>
      </td>
      <td align="center">
        <img src="assets/vdn_ref2va_preview.webp" width="100%" alt="VDN-H3 Preview">
        <br>
        <a href="assets/vdn_ref2va_8step_213s.mp4"><b>▶ Download Full 1344x768 Video (MP4)</b></a>
      </td>
    </tr>
    <tr>
      <td>
        <ul>
          <li><b>Sampling:</b> 4 steps (<code>euler</code> / <code>simple</code>)</li>
          <li><b>Attention:</b> 75% video sparsity top-k (sparse DiT)</li>
          <li><b>Speedup:</b> <b>2.24x faster</b> than VDN; <b>9x faster</b> than native H3</li>
          <li><b>Peak VRAM:</b> ~13.5 GB</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Sampling:</b> 8 steps (<code>er_sde</code> / <code>beta</code>)</li>
          <li><b>Attention:</b> Hybrid (windowed dense + linear delta state)</li>
          <li><b>Speedup:</b> ~3.5x faster than native H3</li>
          <li><b>Peak VRAM:</b> ~18.5 GB</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## 📊 Comprehensive Performance Comparison (RTX 4090)

| Pipeline | Attention Mechanism | Steps / Sampler | 1st Run Latency | Warm Wall Time | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MiniMax H3 Native Dense** | Dense Global Softmax | 50 steps / Euler | ~650s | **~650s (10.8 min)** | ~22.5 GB |
| **H3 Turbo LoRA (Dense)** | Dense Global Softmax | 4 steps / Euler | ~210s | **~190s (3.1 min)** | ~21.0 GB |
| **Video Delta Net (VDN-H3)** | Windowed + Linear Delta | 8 steps / er_sde | 213.0s | **~135s (2.2 min)** | ~18.5 GB |
| **Ref2VA VSA (This Repo)** 🚀 | **75% Video Sparsity + Dense Prefix** | **4 steps / Euler** | **95.2s** | **~72 seconds** | **~13.5 GB** |

---

## 🧠 Technical Architecture: Why Ref2VA was Hard for VSA

[FastVideo](https://github.com/hao-ai-lab/FastVideo) pioneered Visual Sparse Attention (VSA) for text-to-video (T2V) by clustering video tokens into 3D spatial-temporal tiles (4×4×4) and dynamically pruning 75–90% of tiles via a learned gating projection (`to_gate_compress`).

However, **Ref2VA (Reference-to-Video)** was widely considered incompatible with VSA because:
1. Ref2VA prepends dynamic multimodal condition segments (reference image latents, reference audio latents, and text prompt tokens) ahead of the generated video sequence.
2. Naive 3D tiling causes tokens from different modalities to straddle the same tile, corrupting the conditioning masks and breaking identity fidelity.

### The Solution: `Ref2VAVSAGatePatch`
Our patch resolves this with an engineered two-tier attention layout:
1. **Multimodal Dense-Exempt Prefix**: The geometry mapper isolates text, reference image, and reference audio tokens into segment-pure tiles that are **completely exempt from top-k pruning**. Reference tokens always remain 100% dense, guaranteeing strict character identity adherence.
2. **Video-Only Sparse Attention**: VSA top-k pruning is applied *strictly* to generated-video key tiles.
3. **Gate Transplant**: The 50 learned `to_gate_compress` projection matrices are transplanted directly onto the Ref2VA base model (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`).

---

## 📦 Requirements

- **ComfyUI** (latest version with native MiniMax-H3 support).
- **comfy-kitchen** installed with CUDA `sol_attn` support.
- PyTorch 2.4+ and CUDA 12.1+.
- NVIDIA GPU with 24GB VRAM (RTX 3090, RTX 4090, A5000, L40S, etc.).

### Checkpoints
Place the following in your ComfyUI models directory:
- **Diffusion Model**: `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (in `models/diffusion_models/`)
- **Text Encoder**: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (in `models/text_encoders/`)
- **Video VAE**: `minimax_h3_video_vae_fp16.safetensors` (in `models/vae/`)
- **Audio VAE**: `minimax_h3_audio_vae_fp32.safetensors` (in `models/vae/`)
- **Turbo LoRA**: `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (in `models/loras/`)
- **VSA Gate**: `fasth3_vsa_gate.safetensors` (in `models/loras/`)

---

## 🚀 Quick Start

1. Clone or copy this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Kablex/ComfyUI-Ref2VA-VSA
   ```

2. Copy the example character image into your ComfyUI input folder:
   ```bash
   cp ComfyUI-Ref2VA-VSA/assets/example_character.jpg ComfyUI/input/
   ```

3. Restart ComfyUI. The node appears under:
   `FastH3/VSA -> Ref2VA VSA Gate Transplant (EXPERIMENTAL)`

4. Load the ready-to-use workflow from `workflows/ref2va_vsa_4step_rtx4090.json` and click **Queue Prompt**.

---

## 🛠️ Node Wiring

```
[UNETLoader (Ref2VA INT8)]
        │
        ▼
[LoraLoaderModelOnly (Turbo 4-Step LoRA, strength=1.0)]
        │
        ▼
[Ref2VAVSAGatePatch (fasth3_vsa_gate.safetensors, sparsity=0.75)]
        │
        ▼
[MiniMaxH3SigmaShift (shift_video=12, shift_audio=3)]
        │
        ├──► [BasicScheduler (steps=4, scheduler=simple)]
        │
        └──► [BasicGuider] ──► [SamplerCustomAdvanced (sampler=euler)]
```

---

## 🧰 Extracting the Gate File (`tools/extract_vsa_gate.py`)

If you have a FastH3 VSA base checkpoint (`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`), you can extract `fasth3_vsa_gate.safetensors` yourself:

```bash
python tools/extract_vsa_gate.py \
    --input /path/to/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors \
    --output /path/to/ComfyUI/models/loras/fasth3_vsa_gate.safetensors
```

---

## 📄 License & Acknowledgements

- Licensed under the [Apache License 2.0](LICENSE).
- Special thanks to the **FastVideo** team for the VSA-H3 attention concept and to the **ComfyUI** / **comfy-kitchen** developers for the CUDA Sol-Attention kernel primitives.
