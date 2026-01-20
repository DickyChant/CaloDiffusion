#!/usr/bin/env python
"""Inspect a checkpoint file to determine model architecture."""
from __future__ import print_function
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint.pth>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print("Loading checkpoint:", ckpt_path)

ckpt = torch.load(ckpt_path, map_location='cpu')

if isinstance(ckpt, dict):
    print("Checkpoint keys:", list(ckpt.keys()))
    state_dict = ckpt.get('model_state_dict', ckpt)
else:
    state_dict = ckpt

print("")
print("Layer names (first 30):")
for k in list(state_dict.keys())[:30]:
    print("  " + k)

print("")
print("Total parameters:", len(state_dict))

# Check for specific patterns
keys_str = str(list(state_dict.keys()))
has_mfdit = 'r_layers' in keys_str or 'time_layers' in keys_str
has_condunet = 'downs.' in keys_str or 'ups.' in keys_str
has_patch = 'patch_embedder' in keys_str
has_blocks = 'blocks.' in keys_str

print("")
print("Model type detection:")
print("  Has MFDiT patterns (r_layers, time_layers): {}".format(has_mfdit))
print("  Has CondUnet patterns (downs, ups): {}".format(has_condunet))
print("  Has DiT patch_embedder: {}".format(has_patch))
print("  Has DiT blocks: {}".format(has_blocks))

if has_mfdit and has_patch:
    print("")
    print(">>> Detected: MeanFlowDiT (MFAttn)")
elif has_condunet:
    print("")
    print(">>> Detected: CondUnet")
elif has_patch and has_blocks:
    print("")
    print(">>> Detected: PureDiT or similar DiT")
else:
    print("")
    print(">>> Unknown model type")
