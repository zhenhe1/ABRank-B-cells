#!/bin/bash
# Phase 1 Job 1: Large model + source normalization on GPU 0
# Run with: bash run_phase1_job1.sh

echo "=========================================="
echo "Starting Phase 1 Job 1 - GPU 0"
echo "Config: Large model (1024-512-256)"
echo "Learning rate: 2e-4"
echo "Dropout: 0.2"
echo "=========================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training
python waffle/train-abrank-regression.py \
  --config-name=train-abrank-regression-mlp-phase1-job1

echo "=========================================="
echo "Phase 1 Job 1 Complete!"
echo "=========================================="
