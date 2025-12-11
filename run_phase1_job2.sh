#!/bin/bash
# Phase 1 Job 2: Large model + source normalization + tuned hyperparams on GPU 1
# Run with: bash run_phase1_job2.sh

echo "=========================================="
echo "Starting Phase 1 Job 2 - GPU 1"
echo "Config: Large model (1024-512-256)"
echo "Learning rate: 5e-4 (higher)"
echo "Dropout: 0.3 (higher)"
echo "Weight decay: 1e-4"
echo "=========================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Run training
python waffle/train-abrank-regression.py \
  --config-name=train-abrank-regression-mlp-phase1-job2

echo "=========================================="
echo "Phase 1 Job 2 Complete!"
echo "=========================================="
