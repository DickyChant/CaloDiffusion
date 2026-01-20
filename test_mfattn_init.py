#!/usr/bin/env python
"""Test script to verify MFAttn/MFDiT inference support."""
from __future__ import print_function
import json
import sys

# Load config with MFAttn
config_path = "/global/homes/s/sqian/sqian_dev/config_HGCal_photons_sbatch_meanflow.json"
with open(config_path) as f:
    config = json.load(f)

# Override with supported loss type for testing
config["TRAINING_OBJ"] = "hybrid_weight"

print("SHOWER_EMBED: {}".format(config.get('SHOWER_EMBED')))

from calodiffusion.models.calodiffusion import CaloDiffusion

model = CaloDiffusion(config, n_steps=400, loss_type="l2")
print("Model type: {}".format(type(model.model).__name__))
print("is_meanflow_dit: {}".format(model.is_meanflow_dit))

# Expected results
expected_model = "MeanFlowDiT"
actual_model = type(model.model).__name__

if actual_model == expected_model and model.is_meanflow_dit:
    print("SUCCESS: CaloDiffusion correctly initialized with MFAttn!")
    sys.exit(0)
else:
    print("FAILED: Expected {}, got {}".format(expected_model, actual_model))
    sys.exit(1)

