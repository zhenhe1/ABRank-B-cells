"""
Minimal test script for MLP model instantiation.

This script creates minimal synthetic data to test that the RegressionMLPAbAgIntLM
model can be instantiated and performs a forward pass without errors.
"""

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from waffle.models.abrank_regression_mlp import RegressionMLPAbAgIntLM


def create_minimal_batch(batch_size=2, ab_nodes=10, ag_nodes=15):
    """
    Create a minimal batch with synthetic data for testing.

    Args:
        batch_size: Number of samples in batch
        ab_nodes: Number of antibody residues per sample
        ag_nodes: Number of antigen residues per sample

    Returns:
        Batch object with synthetic data
    """
    data_list = []

    for i in range(batch_size):
        # Create synthetic embeddings
        x_b = torch.randn(ab_nodes, 512)  # AntiBERTy embeddings
        x_g = torch.randn(ag_nodes, 1280)  # ESM-2 embeddings

        # Create a synthetic label (binding affinity in log scale)
        y = torch.randn(1) * 2  # Random affinity between roughly -2 to 2

        # Create a data object
        data = Data(
            x_b=x_b,
            x_g=x_g,
            y=y,
            name=f"test_sample_{i}"
        )
        data_list.append(data)

    # Create batch (this automatically creates x_b_batch and x_g_batch)
    batch = Batch.from_data_list(data_list)

    # Manually create batch indices for x_b and x_g
    x_b_batch = torch.cat([torch.full((ab_nodes,), i) for i in range(batch_size)])
    x_g_batch = torch.cat([torch.full((ag_nodes,), i) for i in range(batch_size)])

    batch.x_b_batch = x_b_batch
    batch.x_g_batch = x_g_batch

    return batch


def create_minimal_config():
    """
    Create a minimal configuration for the MLP model.

    Returns:
        OmegaConf DictConfig with minimal required settings
    """
    config_dict = {
        "encoder": {
            "ab": {
                "input_dim": 512  # AntiBERTy embedding dimension
            },
            "ag": {
                "input_dim": 1280  # ESM-2 embedding dimension
            }
        },
        "regressor": {
            "hidden_dims": [512, 256],
            "dropout": 0.2
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-4
        }
    }

    return OmegaConf.create(config_dict)


def test_model_instantiation():
    """Test model instantiation and forward pass."""
    print("=" * 60)
    print("Testing MLP Model Instantiation")
    print("=" * 60)

    # Create minimal config
    print("\n1. Creating minimal configuration...")
    cfg = create_minimal_config()
    print("   ✓ Configuration created")
    print(f"   - AB embedding dim: {cfg.encoder.ab.input_dim}")
    print(f"   - AG embedding dim: {cfg.encoder.ag.input_dim}")
    print(f"   - Hidden dims: {cfg.regressor.hidden_dims}")
    print(f"   - Dropout: {cfg.regressor.dropout}")

    # Instantiate model
    print("\n2. Instantiating model...")
    model = RegressionMLPAbAgIntLM(cfg=cfg)
    print("   ✓ Model instantiated successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")

    # Create minimal batch
    print("\n3. Creating minimal batch...")
    batch = create_minimal_batch(batch_size=2, ab_nodes=10, ag_nodes=15)
    print("   ✓ Batch created")
    print(f"   - Batch size: 2")
    print(f"   - AB nodes per sample: 10")
    print(f"   - AG nodes per sample: 15")
    print(f"   - x_b shape: {batch.x_b.shape}")
    print(f"   - x_g shape: {batch.x_g.shape}")
    print(f"   - y shape: {batch.y.shape}")

    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print("   ✓ Forward pass successful")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output values: {output.squeeze().tolist()}")
    print(f"   - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test with different batch sizes
    print("\n5. Testing with different batch sizes...")
    for bs in [1, 4, 8]:
        batch_test = create_minimal_batch(batch_size=bs, ab_nodes=20, ag_nodes=30)
        with torch.no_grad():
            output_test = model(batch_test)
        print(f"   ✓ Batch size {bs}: output shape {output_test.shape}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nModel is ready for training with real data.")


if __name__ == "__main__":
    test_model_instantiation()
