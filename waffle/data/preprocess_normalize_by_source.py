"""
Preprocess training data to normalize affinity values by source database.

This script:
1. Loads the training registry
2. Computes mean and std for each source database
3. Creates a normalized version of the registry with source-normalized logAff values
4. Saves normalization statistics for later use during evaluation

Author: Claude Code
Date: 2025-12-10
"""

import os
from pathlib import Path

import pandas as pd
from loguru import logger


def normalize_by_source(data_path: str, output_suffix: str = "normalized") -> None:
    """
    Normalize logAff values by source database.

    Args:
        data_path: Path to data directory containing AbRank dataset
        output_suffix: Suffix to add to output files
    """
    # Paths
    registry_path = Path(data_path) / "AbRank" / "processed" / "registry-regression" / "AbRank-regression-all.csv"
    train_split_path = Path(data_path) / "AbRank" / "processed" / "splits-regression" / "Split_AF3" / "balanced-train-regression.csv"

    output_dir = Path(data_path) / "AbRank" / "processed" / "splits-regression" / "Split_AF3"
    output_train_path = output_dir / f"balanced-train-regression-{output_suffix}.csv"
    output_stats_path = output_dir / f"source-normalization-stats.csv"

    logger.info(f"Loading registry from: {registry_path}")
    registry = pd.read_csv(registry_path)

    logger.info(f"Loading training split from: {train_split_path}")
    train_split = pd.read_csv(train_split_path)

    # Filter to training samples only
    train_registry = registry[registry['setType'] == 'train'].copy()

    logger.info(f"Training samples: {len(train_registry)}")
    logger.info(f"Source databases: {train_registry['srcDB'].nunique()}")

    # Compute normalization statistics per source
    normalization_stats = []

    for source in train_registry['srcDB'].unique():
        source_data = train_registry[train_registry['srcDB'] == source]
        mean = source_data['logAff'].mean()
        std = source_data['logAff'].std()
        count = len(source_data)

        normalization_stats.append({
            'srcDB': source,
            'mean': mean,
            'std': std,
            'count': count,
            'min': source_data['logAff'].min(),
            'max': source_data['logAff'].max(),
        })

        logger.info(f"  {source}: mean={mean:.4f}, std={std:.4f}, count={count}")

        # Normalize
        train_registry.loc[train_registry['srcDB'] == source, 'logAff_normalized'] = \
            (source_data['logAff'] - mean) / std

    # Save normalization statistics
    stats_df = pd.DataFrame(normalization_stats)
    stats_df.to_csv(output_stats_path, index=False)
    logger.info(f"Saved normalization statistics to: {output_stats_path}")

    # Create normalized training split
    # Join with normalized registry to get normalized values
    train_split_normalized = train_split.merge(
        train_registry[['dbID', 'logAff_normalized']],
        on='dbID',
        how='left'
    )

    # Replace logAff with normalized values
    train_split_normalized['logAff'] = train_split_normalized['logAff_normalized']
    train_split_normalized = train_split_normalized.drop(columns=['logAff_normalized'])

    # Save normalized training split
    train_split_normalized.to_csv(output_train_path, index=False)
    logger.info(f"Saved normalized training split to: {output_train_path}")

    # Report statistics
    logger.info("\n" + "="*70)
    logger.info("NORMALIZATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Original logAff range: [{train_registry['logAff'].min():.4f}, {train_registry['logAff'].max():.4f}]")
    logger.info(f"Normalized logAff range: [{train_registry['logAff_normalized'].min():.4f}, {train_registry['logAff_normalized'].max():.4f}]")
    logger.info(f"Original logAff mean: {train_registry['logAff'].mean():.4f}")
    logger.info(f"Normalized logAff mean: {train_registry['logAff_normalized'].mean():.4f}")
    logger.info(f"Original logAff std: {train_registry['logAff'].std():.4f}")
    logger.info(f"Normalized logAff std: {train_registry['logAff_normalized'].std():.4f}")
    logger.info("="*70)

    # Show per-source statistics
    logger.info("\nPer-Source Statistics:")
    print(stats_df.to_string(index=False))

    return output_train_path, output_stats_path


if __name__ == "__main__":
    # Get data path from environment
    data_path = os.environ.get('DATA_PATH')
    if not data_path:
        raise ValueError("DATA_PATH environment variable not set")

    normalize_by_source(data_path)
