"""
Create Beijing PM2.5 datasets with continuous long paths for test evaluation.

Training/Validation: Short overlapping windows (168 hours, good for training)
Test: Full continuous paths (monthly or entire year, better for evaluation)
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from beijing_pm25_dataset import BeijingPM25Dataset


def create_continuous_test_paths(
    df: pd.DataFrame,
    loader: BeijingPM25Dataset,
    noise_level: float,
    obs_rate: float,
    path_type: str = 'monthly',
    seed: int = 42
) -> dict:
    """
    Create continuous long paths for test evaluation.

    Args:
        df: Test DataFrame (2014 year)
        loader: BeijingPM25Dataset instance
        noise_level: Noise coefficient ζ
        obs_rate: Observation rate
        path_type: 'monthly' (12 paths) or 'full' (1 path for entire year)
        seed: Random seed

    Returns:
        Dataset dict with continuous paths
    """
    print(f"\nCreating continuous test paths (type={path_type})...")

    # Add noise
    df_noisy = loader.add_observation_noise(df, noise_level=noise_level, seed=seed)

    # Create irregular observations
    df_irregular = loader.create_irregular_observations(
        df_noisy,
        observation_rate=obs_rate,
        observation_dependent=True,
        seed=seed + 999
    )

    if path_type == 'monthly':
        # Split into 12 monthly paths (no overlap)
        path_length_hours = 730  # ~1 month (30.4 days average)
        stride_hours = 730       # No overlap
    elif path_type == 'quarterly':
        # Split into 4 quarterly paths
        path_length_hours = 2190  # ~3 months (91 days)
        stride_hours = 2190
    elif path_type == 'full':
        # Entire year as one path
        path_length_hours = len(df)
        stride_hours = len(df)
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    dataset = loader.create_path_batches(
        df_irregular,
        path_length_hours=path_length_hours,
        stride_hours=stride_hours
    )

    # Add metadata
    dataset['delta_t'] = 0.001
    dataset['maturity'] = 1.0
    dataset['path_type'] = path_type
    dataset['noise_level'] = noise_level

    return dataset


def main():
    """Create datasets with continuous test paths."""

    print("="*70)
    print("Creating Beijing PM2.5 Datasets with Continuous Test Paths")
    print("="*70)

    data_path = "../data/PRSA_data_2010.1.1-2014.12.31.csv"
    output_dir = Path("../data/")

    # Load and preprocess
    loader = BeijingPM25Dataset(data_path)
    print("\nLoading and preprocessing data...")
    loader.load_and_preprocess()

    # Split by year
    print("\nSplitting by year...")
    train_df, val_df, test_df = loader.train_val_test_split()

    # Noise levels to test
    noise_levels = [0.25, 0.5, 1.0]
    obs_rate = 0.5

    for noise_level in noise_levels:
        print(f"\n{'='*70}")
        print(f"Processing noise level ζ={noise_level}")
        print(f"{'='*70}")

        # ==============================================================
        # TRAINING: Short overlapping windows (good for learning)
        # ==============================================================
        print("\n[TRAIN] Creating short overlapping windows...")
        df_train_noisy = loader.add_observation_noise(
            train_df, noise_level=noise_level, seed=42
        )
        df_train_irregular = loader.create_irregular_observations(
            df_train_noisy,
            observation_rate=obs_rate,
            observation_dependent=True,
            seed=42
        )
        dataset_train = loader.create_path_batches(
            df_train_irregular,
            path_length_hours=168,  # 1 week
            stride_hours=24         # Overlap for more training data
        )
        dataset_train['delta_t'] = 0.001
        dataset_train['maturity'] = 1.0

        # Save train
        train_path = output_dir / f"BeijingPM25_continuous_noise{noise_level}_train.pkl"
        loader.save_dataset(dataset_train, str(train_path))

        # ==============================================================
        # VALIDATION: Short overlapping windows
        # ==============================================================
        print("\n[VAL] Creating short overlapping windows...")
        df_val_noisy = loader.add_observation_noise(
            val_df, noise_level=noise_level, seed=42
        )
        df_val_irregular = loader.create_irregular_observations(
            df_val_noisy,
            observation_rate=obs_rate,
            observation_dependent=True,
            seed=43
        )
        dataset_val = loader.create_path_batches(
            df_val_irregular,
            path_length_hours=168,
            stride_hours=24
        )
        dataset_val['delta_t'] = 0.001
        dataset_val['maturity'] = 1.0

        # Save val
        val_path = output_dir / f"BeijingPM25_continuous_noise{noise_level}_val.pkl"
        loader.save_dataset(dataset_val, str(val_path))

        # ==============================================================
        # TEST: CONTINUOUS LONG PATHS (better evaluation)
        # ==============================================================

        # Option 1: Monthly paths (12 continuous months)
        print("\n[TEST] Creating MONTHLY continuous paths...")
        dataset_test_monthly = create_continuous_test_paths(
            test_df, loader, noise_level, obs_rate,
            path_type='monthly', seed=44
        )
        test_monthly_path = output_dir / f"BeijingPM25_continuous_noise{noise_level}_test_monthly.pkl"
        loader.save_dataset(dataset_test_monthly, str(test_monthly_path))

        # Option 2: Quarterly paths (4 continuous quarters)
        print("\n[TEST] Creating QUARTERLY continuous paths...")
        dataset_test_quarterly = create_continuous_test_paths(
            test_df, loader, noise_level, obs_rate,
            path_type='quarterly', seed=44
        )
        test_quarterly_path = output_dir / f"BeijingPM25_continuous_noise{noise_level}_test_quarterly.pkl"
        loader.save_dataset(dataset_test_quarterly, str(test_quarterly_path))

        # Also keep the standard overlapping test set for comparison
        print("\n[TEST] Creating standard overlapping windows (for comparison)...")
        df_test_noisy = loader.add_observation_noise(
            test_df, noise_level=noise_level, seed=42
        )
        df_test_irregular = loader.create_irregular_observations(
            df_test_noisy,
            observation_rate=obs_rate,
            observation_dependent=True,
            seed=44
        )
        dataset_test_standard = loader.create_path_batches(
            df_test_irregular,
            path_length_hours=168,
            stride_hours=24
        )
        dataset_test_standard['delta_t'] = 0.001
        dataset_test_standard['maturity'] = 1.0
        test_standard_path = output_dir / f"BeijingPM25_continuous_noise{noise_level}_test.pkl"
        loader.save_dataset(dataset_test_standard, str(test_standard_path))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nDataset structure:")
    print("  TRAIN/VAL: 168-hour overlapping windows (efficient training)")
    print("  TEST Options:")
    print("    - Standard: 168-hour overlapping windows (comparable to train)")
    print("    - Monthly: ~730-hour continuous paths (12 months)")
    print("    - Quarterly: ~2190-hour continuous paths (4 quarters)")
    print("\nTest evaluation recommendations:")
    print("  - Use MONTHLY for realistic long-term prediction evaluation")
    print("  - Use QUARTERLY for very long-term extrapolation tests")
    print("  - Compare with Standard to see effect of continuous evaluation")
    print("="*70)


if __name__ == "__main__":
    main()
