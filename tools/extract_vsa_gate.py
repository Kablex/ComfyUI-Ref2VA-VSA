#!/usr/bin/env python3
"""
Extract 50 FastH3 VSA coarse-gate matrices (`to_gate_compress`) from a FastH3
checkpoint into a standalone `fasth3_vsa_gate.safetensors` file.

Usage:
    python extract_vsa_gate.py \
        --input /path/to/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors \
        --output /path/to/models/loras/fasth3_vsa_gate.safetensors
"""

import argparse
import re
import sys
import torch
import safetensors.torch


def extract_gates(input_path: str, output_path: str):
    print(f"Opening input checkpoint: {input_path}")
    gate_tensors = {}
    
    pattern = re.compile(r"^(?:diffusion_model\.)?blocks\.(\d+)\.attn\.to_gate_compress\.weight$")
    alt_pattern = re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.(?:weight|set_weight)$")

    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Total keys in checkpoint: {len(keys)}")
        
        for k in keys:
            m = pattern.match(k) or alt_pattern.match(k)
            if m:
                block_idx = int(m.group(1))
                tensor = f.get_tensor(k)
                target_key = f"blocks.{block_idx}.attn.to_gate_compress.weight"
                gate_tensors[target_key] = tensor.contiguous()

    print(f"Found {len(gate_tensors)} to_gate_compress gate matrices.")
    if len(gate_tensors) != 50:
        print(f"WARNING: Expected 50 gate matrices, found {len(gate_tensors)}!")
        if len(gate_tensors) == 0:
            sys.exit(1)

    print(f"Saving extracted gates to: {output_path}")
    safetensors.torch.save_file(gate_tensors, output_path)
    print("Done! Extracted gate file is ready to use with Ref2VAVSAGatePatch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FastH3 VSA gate matrices.")
    parser.add_argument("--input", "-i", required=True, help="Input FastH3 VSA safetensors checkpoint")
    parser.add_argument("--output", "-o", default="fasth3_vsa_gate.safetensors", help="Output safetensors path")
    args = parser.parse_args()

    extract_gates(args.input, args.output)
