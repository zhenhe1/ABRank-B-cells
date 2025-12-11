"""
Ranking-Based Evaluation Script for MLP Regression Model

This script evaluates a trained MLP checkpoint using pairwise ranking metrics:
- AUC calculation via pairwise comparisons (primary metric)
- Spearman rank correlation
- Pairwise comparison accuracy
- Visualizations (ROC curves, ranking plots)

Unlike the regression evaluation, this script works with pairwise comparison files
and does not require absolute ground truth affinity values.

Usage:
    python waffle/evaluate_mlp_ranking.py --checkpoint path/to/checkpoint.ckpt

Author: Claude Code
Date: 2025-12-10
"""

# Standard library imports
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr
from torch import Tensor
from torchmetrics.classification import BinaryAUROC
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm

# Project imports
from waffle.data.abrank_regression_datamodule import AbRankDataModule
from waffle.models.abrank_regression_mlp import RegressionMLPAbAgIntLM

# Configure visualization theme
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("Set2")


# ============================================================================
# Phase 1: Core Infrastructure
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained MLP model using pairwise ranking metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python waffle/evaluate_mlp_ranking.py --checkpoint runs/.../epoch_094.ckpt

  # Custom output directory
  python waffle/evaluate_mlp_ranking.py --checkpoint runs/.../epoch_094.ckpt --output-dir results/eval

  # CPU-only evaluation
  python waffle/evaluate_mlp_ranking.py --checkpoint runs/.../epoch_094.ckpt --device cpu
        """
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )

    # Optional arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data directory (defaults to DATA_PATH env var)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to checkpoint_dir/evaluation_ranking)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for evaluation (default: auto-detect)"
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["generalization", "perturbation"],
        help="Test splits to evaluate (default: generalization perturbation)"
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device for evaluation."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using {device_arg} device")

    return device


def load_model(checkpoint_path: str, device: torch.device) -> RegressionMLPAbAgIntLM:
    """Load trained MLP model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")

    try:
        model = RegressionMLPAbAgIntLM.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {total_params:,}")

        return model

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


# ============================================================================
# Phase 2: Data Loading
# ============================================================================

def load_pairwise_comparisons(data_path: str, split_name: str) -> pd.DataFrame:
    """
    Load pairwise comparison CSV file for a test split.

    Args:
        data_path: Path to data directory
        split_name: Name of test split (e.g., 'generalization', 'perturbation')

    Returns:
        DataFrame with columns: dbID1, dbID2, deltaLogKD, op, fileName1, fileName2, combName
    """
    split_dir = Path(data_path) / "AbRank" / "processed" / "splits-regression" / "Split_AF3"
    csv_path = split_dir / f"test-{split_name}-swapped.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Pairwise comparison file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} pairwise comparisons for {split_name}")

    return df


def setup_datamodule(data_path: str, batch_size: int = 32) -> AbRankDataModule:
    """Setup datamodule for loading test datasets."""
    logger.info("Setting up datamodule...")

    split_dir = Path(data_path) / "AbRank" / "processed" / "splits-regression" / "Split_AF3"

    train_split_path = str(split_dir / "balanced-train-regression.csv")
    test_split_path_dict = {
        "generalization": str(split_dir / "test-generalization-swapped.csv"),
        "perturbation": str(split_dir / "test-perturbation-swapped.csv"),
    }

    dm = AbRankDataModule(
        root=data_path,
        train_split_path=train_split_path,
        test_split_path_dict=test_split_path_dict,
        seed=42,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    dm.prepare_data()
    dm.setup(stage="test")

    logger.info("Datamodule setup complete")

    return dm


# ============================================================================
# Phase 3: Inference
# ============================================================================

def get_predictions_for_samples(
    model: RegressionMLPAbAgIntLM,
    datamodule: AbRankDataModule,
    dbids: List[int],
    device: torch.device
) -> Dict[int, float]:
    """
    Get model predictions for a list of dbIDs.

    Args:
        model: Trained model
        datamodule: Datamodule with dataset access
        dbids: List of database IDs to get predictions for
        device: Device to run inference on

    Returns:
        Dictionary mapping dbID to predicted score
    """
    model.eval()
    predictions = {}

    # Get the base dataset
    dataset = datamodule.dataset
    dbID2idx = datamodule.dbID2idx

    # Get predictions for each dbID
    with torch.no_grad():
        for dbid in tqdm(dbids, desc="Getting predictions"):
            if dbid not in dbID2idx:
                logger.warning(f"dbID {dbid} not found in dataset")
                predictions[dbid] = float('nan')
                continue

            # Get the data
            idx = dbID2idx[dbid]
            data = dataset.get(idx)

            # Create a batch with single sample, following the same batching as dataloader
            batch = PyGBatch.from_data_list(
                [data],
                follow_batch=["x_b", "x_g"],
                exclude_keys=["metadata", "edge_index_bg", "y_b", "y_g"]
            )
            batch = batch.to(device)

            # Get prediction
            pred = model(batch)
            predictions[dbid] = pred.squeeze().cpu().item()

    return predictions


# ============================================================================
# Phase 4: Ranking Metrics
# ============================================================================

def calculate_pairwise_ranking_metrics(
    predictions: Dict[int, float],
    pairwise_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate ranking metrics from pairwise comparisons.

    Args:
        predictions: Dictionary mapping dbID to predicted score
        pairwise_df: DataFrame with pairwise comparisons

    Returns:
        Dictionary of metric names to values
    """
    # Prepare lists for binary classification
    pred_labels = []
    true_labels = []
    pred_diffs = []
    true_diffs = []

    valid_pairs = 0
    skipped_pairs = 0

    for _, row in pairwise_df.iterrows():
        dbid1 = row['dbID1']
        dbid2 = row['dbID2']
        true_delta = row['deltaLogKD']  # logAff(dbID1) - logAff(dbID2)

        # Check if predictions are available
        if dbid1 not in predictions or dbid2 not in predictions:
            skipped_pairs += 1
            continue

        pred1 = predictions[dbid1]
        pred2 = predictions[dbid2]

        # Skip if predictions are NaN
        if np.isnan(pred1) or np.isnan(pred2):
            skipped_pairs += 1
            continue

        # Predicted difference
        pred_delta = pred1 - pred2

        # Binary labels for AUROC: 1 if first > second, 0 otherwise
        # Predicted: based on predicted scores
        pred_label = 1.0 if pred1 > pred2 else 0.0
        # True: based on true deltaLogKD
        true_label = 1.0 if true_delta > 0 else 0.0

        pred_labels.append(pred_label)
        true_labels.append(true_label)
        pred_diffs.append(pred_delta)
        true_diffs.append(true_delta)
        valid_pairs += 1

    if skipped_pairs > 0:
        logger.warning(f"Skipped {skipped_pairs} pairs due to missing predictions")

    logger.info(f"Using {valid_pairs} valid pairs for evaluation")

    if valid_pairs == 0:
        logger.error("No valid pairs for evaluation!")
        return {
            'auc': float('nan'),
            'accuracy': float('nan'),
            'spearman': float('nan'),
            'num_pairs': 0,
            'num_correct': 0,
        }

    # Calculate AUROC
    pred_labels_tensor = torch.tensor(pred_labels)
    true_labels_tensor = torch.tensor(true_labels)
    auroc = BinaryAUROC()
    auc_score = auroc(pred_labels_tensor, true_labels_tensor).item()

    # Calculate accuracy (percentage of correctly ordered pairs)
    num_correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = num_correct / valid_pairs

    # Calculate Spearman correlation on the differences
    spearman_corr, _ = spearmanr(true_diffs, pred_diffs)

    metrics = {
        'auc': auc_score,
        'accuracy': accuracy,
        'spearman': spearman_corr,
        'num_pairs': valid_pairs,
        'num_correct': num_correct,
    }

    return metrics, pred_labels, true_labels, pred_diffs, true_diffs


# ============================================================================
# Phase 5: Visualizations
# ============================================================================

def plot_roc_curve(
    pred_labels: List[float],
    true_labels: List[float],
    split_name: str,
    output_dir: Path,
    auc_score: float
) -> Path:
    """Plot ROC curve for pairwise ranking predictions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)

    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'ROC Curve - Pairwise Rankings - {split_name}', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path_png = output_dir / f"{split_name}_roc_curve.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    output_path_pdf = output_dir / f"{split_name}_roc_curve.pdf"
    fig.savefig(output_path_pdf, bbox_inches='tight')

    plt.close(fig)

    return output_path_png


def plot_ranking_correlation(
    pred_diffs: List[float],
    true_diffs: List[float],
    split_name: str,
    output_dir: Path,
    spearman: float
) -> Path:
    """Plot correlation between predicted and true rank differences."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(true_diffs, pred_diffs, alpha=0.3, s=20)

    # Add regression line
    z = np.polyfit(true_diffs, pred_diffs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(true_diffs), max(true_diffs), 100)
    ax.plot(x_line, p(x_line), "r--", lw=2, label=f'Linear fit (y={z[0]:.2f}x+{z[1]:.2f})')

    # Add perfect correlation line
    min_val = min(min(true_diffs), min(pred_diffs))
    max_val = max(max(true_diffs), max(pred_diffs))
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, alpha=0.5, label='Perfect correlation')

    ax.set_xlabel('True Rank Difference (ΔlogKd)', fontsize=14)
    ax.set_ylabel('Predicted Rank Difference', fontsize=14)
    ax.set_title(f'Ranking Correlation - {split_name}\nSpearman ρ = {spearman:.3f}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path_png = output_dir / f"{split_name}_ranking_correlation.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    output_path_pdf = output_dir / f"{split_name}_ranking_correlation.pdf"
    fig.savefig(output_path_pdf, bbox_inches='tight')

    plt.close(fig)

    return output_path_png


def plot_prediction_distribution(
    predictions: Dict[int, float],
    split_name: str,
    output_dir: Path
) -> Path:
    """Plot distribution of predicted scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pred_values = [v for v in predictions.values() if not np.isnan(v)]

    ax.hist(pred_values, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(pred_values), color='r', linestyle='--', lw=2,
               label=f'Mean = {np.mean(pred_values):.3f}')
    ax.axvline(np.median(pred_values), color='g', linestyle='--', lw=2,
               label=f'Median = {np.median(pred_values):.3f}')

    ax.set_xlabel('Predicted Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Distribution of Predicted Scores - {split_name}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path_png = output_dir / f"{split_name}_prediction_distribution.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    output_path_pdf = output_dir / f"{split_name}_prediction_distribution.pdf"
    fig.savefig(output_path_pdf, bbox_inches='tight')

    plt.close(fig)

    return output_path_png


# ============================================================================
# Phase 6: Export Functions
# ============================================================================

def export_predictions_csv(
    predictions: Dict[int, float],
    split_name: str,
    output_dir: Path
) -> Path:
    """Export predictions to CSV."""
    df = pd.DataFrame([
        {'dbID': dbid, 'predicted_score': score}
        for dbid, score in sorted(predictions.items())
    ])

    output_path = output_dir / f"{split_name}_predictions.csv"
    df.to_csv(output_path, index=False)

    return output_path


def export_pairwise_results_csv(
    pairwise_df: pd.DataFrame,
    predictions: Dict[int, float],
    split_name: str,
    output_dir: Path
) -> Path:
    """Export pairwise comparison results to CSV."""
    results = []

    for _, row in pairwise_df.iterrows():
        dbid1 = row['dbID1']
        dbid2 = row['dbID2']
        true_delta = row['deltaLogKD']

        pred1 = predictions.get(dbid1, float('nan'))
        pred2 = predictions.get(dbid2, float('nan'))
        pred_delta = pred1 - pred2 if not (np.isnan(pred1) or np.isnan(pred2)) else float('nan')

        # Determine if ranking is correct
        if not np.isnan(pred_delta):
            pred_order_correct = (pred_delta > 0) == (true_delta > 0)
        else:
            pred_order_correct = None

        results.append({
            'dbID1': dbid1,
            'dbID2': dbid2,
            'true_delta': true_delta,
            'pred1': pred1,
            'pred2': pred2,
            'pred_delta': pred_delta,
            'correct_order': pred_order_correct,
        })

    df = pd.DataFrame(results)
    output_path = output_dir / f"{split_name}_pairwise_results.csv"
    df.to_csv(output_path, index=False)

    return output_path


def export_metrics_json(
    all_results: Dict[str, Dict],
    output_dir: Path,
    checkpoint_path: str
) -> Path:
    """Export aggregated metrics to JSON."""
    summary = {
        'checkpoint': str(checkpoint_path),
        'evaluation_time': str(pd.Timestamp.now()),
        'evaluation_mode': 'ranking_only',
        'results_by_split': all_results,
        'summary_statistics': {
            'avg_auc': float(np.mean([r['auc'] for r in all_results.values()])),
            'avg_accuracy': float(np.mean([r['accuracy'] for r in all_results.values()])),
            'avg_spearman': float(np.mean([r['spearman'] for r in all_results.values()])),
        }
    }

    output_path = output_dir / "evaluation_metrics_ranking.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return output_path


# ============================================================================
# Phase 7: Main Integration
# ============================================================================

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get device
    device = get_device(args.device)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "evaluation_ranking"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get data path
    data_path = args.data_path or os.environ.get('DATA_PATH')
    if not data_path:
        raise ValueError("Data path must be provided via --data-path or DATA_PATH environment variable")

    # Load model
    model = load_model(str(checkpoint_path), device)

    # Setup datamodule
    dm = setup_datamodule(data_path, batch_size=args.batch_size)

    # Evaluate on all requested splits
    all_results = {}

    for split_name in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {split_name}")
        logger.info(f"{'='*60}")

        # Load pairwise comparisons
        pairwise_df = load_pairwise_comparisons(data_path, split_name)

        # Get unique dbIDs
        unique_dbids = sorted(set(pairwise_df['dbID1'].unique()) | set(pairwise_df['dbID2'].unique()))
        logger.info(f"Found {len(unique_dbids)} unique samples in pairwise comparisons")

        # Get predictions for all samples
        predictions = get_predictions_for_samples(model, dm, unique_dbids, device)

        # Calculate ranking metrics
        metrics, pred_labels, true_labels, pred_diffs, true_diffs = calculate_pairwise_ranking_metrics(
            predictions, pairwise_df
        )

        # Print metrics
        logger.info(f"\nRanking Metrics for {split_name}:")
        logger.info(f"  AUC (pairwise):        {metrics['auc']:.4f}")
        logger.info(f"  Accuracy (pairwise):   {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_pairs']})")
        logger.info(f"  Spearman correlation:  {metrics['spearman']:.4f}")

        # Generate visualizations
        if not args.no_visualizations:
            logger.info(f"\nGenerating visualizations for {split_name}...")
            plot_roc_curve(pred_labels, true_labels, split_name, output_dir, metrics['auc'])
            logger.info("  ✓ ROC curve saved")

            plot_ranking_correlation(pred_diffs, true_diffs, split_name, output_dir, metrics['spearman'])
            logger.info("  ✓ Ranking correlation plot saved")

            plot_prediction_distribution(predictions, split_name, output_dir)
            logger.info("  ✓ Prediction distribution plot saved")

        # Export results
        logger.info(f"\nExporting results for {split_name}...")
        export_predictions_csv(predictions, split_name, output_dir)
        logger.info("  ✓ Predictions saved to CSV")

        export_pairwise_results_csv(pairwise_df, predictions, split_name, output_dir)
        logger.info("  ✓ Pairwise results saved to CSV")

        all_results[split_name] = metrics

    # Export summary JSON
    summary_path = export_metrics_json(all_results, output_dir, str(checkpoint_path))
    logger.info(f"\n✓ Metrics summary saved to: {summary_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("RANKING EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    for split_name, metrics in all_results.items():
        logger.info(f"\n{split_name}:")
        logger.info(f"  AUC (pairwise):        {metrics['auc']:.4f}")
        logger.info(f"  Accuracy (pairwise):   {metrics['accuracy']:.4f}")
        logger.info(f"  Spearman correlation:  {metrics['spearman']:.4f}")
        logger.info(f"  Valid pairs:           {metrics['num_pairs']}")

    logger.info(f"\n✓ Evaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
