"""
Beijing PM2.5 Dataset Loader and Preprocessor

This module implements data loading and preprocessing for the UCI Beijing PM2.5 dataset,
demonstrating noise-robust loss functions with real-world time series data.

Reference: https://archive.ics.uci.edu/dataset/381/beijing+pm2.5+data
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from scipy.special import expit as sigmoid


class BeijingPM25Dataset:
    """
    Handles loading, preprocessing, and augmentation of Beijing PM2.5 data.

    The dataset contains hourly PM2.5 measurements from 2010-2014 along with
    meteorological variables (temperature, pressure, wind, etc.).
    """

    def __init__(self, data_path: str = "../data/PRSA_data_2010.1.1-2014.12.31.csv"):
        """
        Args:
            data_path: Path to the Beijing PM2.5 CSV file
        """
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_and_preprocess(self,
                           fill_method: str = 'interpolate',
                           remove_extreme_outliers: bool = True) -> pd.DataFrame:
        """
        Load and perform basic preprocessing on the raw CSV data.

        Args:
            fill_method: Method for filling missing PM2.5 values ('interpolate', 'forward', 'drop')
            remove_extreme_outliers: Whether to cap extreme PM2.5 values (>1000)

        Returns:
            Preprocessed DataFrame with datetime index
        """
        # Load CSV
        df = pd.read_csv(self.data_path)

        # Create datetime index from year, month, day, hour columns
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime')

        # Drop the redundant No column
        if 'No' in df.columns:
            df = df.drop(columns=['No'])

        # Handle missing PM2.5 values
        pm25_missing_count = df['pm2.5'].isna().sum()
        print(f"Missing PM2.5 values: {pm25_missing_count} ({100*pm25_missing_count/len(df):.1f}%)")

        if fill_method == 'interpolate':
            df['pm2.5'] = df['pm2.5'].interpolate(method='linear', limit_direction='both')
        elif fill_method == 'forward':
            df['pm2.5'] = df['pm2.5'].fillna(method='ffill').fillna(method='bfill')
        elif fill_method == 'drop':
            df = df.dropna(subset=['pm2.5'])
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")

        # Remove extreme outliers (sensor errors)
        if remove_extreme_outliers:
            outlier_mask = df['pm2.5'] > 1000
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                print(f"Capping {n_outliers} extreme outliers (PM2.5 > 1000)")
                df.loc[outlier_mask, 'pm2.5'] = 1000

        # Fill missing meteorological variables (less critical)
        for col in ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Convert categorical wind direction to one-hot
        if 'cbwd' in df.columns:
            cbwd_dummies = pd.get_dummies(df['cbwd'], prefix='wind', drop_first=False)
            df = pd.concat([df, cbwd_dummies], axis=1)
            df = df.drop(columns=['cbwd'])

        self.df = df
        return df

    def train_val_test_split(self,
                             train_years: List[int] = [2010, 2011, 2012],
                             val_years: List[int] = [2013],
                             test_years: List[int] = [2014]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by year into train/val/test sets.

        Args:
            train_years: Years for training
            val_years: Years for validation
            test_years: Years for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            raise ValueError("Must call load_and_preprocess() first")

        # Split by year
        self.train_df = self.df[self.df['year'].isin(train_years)]
        self.val_df = self.df[self.df['year'].isin(val_years)]
        self.test_df = self.df[self.df['year'].isin(test_years)]

        print(f"Train: {len(self.train_df)} samples ({train_years})")
        print(f"Val: {len(self.val_df)} samples ({val_years})")
        print(f"Test: {len(self.test_df)} samples ({test_years})")

        return self.train_df, self.val_df, self.test_df

    def add_observation_noise(self,
                             df: pd.DataFrame,
                             noise_level: float = 0.5,
                             seed: Optional[int] = None) -> pd.DataFrame:
        """
        Add Gaussian noise to PM2.5 observations: O_t = X_t + ε, ε ~ N(0, ζ·σ_X)

        Args:
            df: DataFrame with 'pm2.5' column (clean latent process)
            noise_level: ζ, noise coefficient (0.25, 0.5, 1.0 typical)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with additional 'pm2.5_observed' column (noisy observations)
        """
        if seed is not None:
            np.random.seed(seed)

        pm25_clean = df['pm2.5'].values
        pm25_std = np.std(pm25_clean)

        noise = np.random.normal(0, noise_level * pm25_std, size=len(pm25_clean))
        pm25_noisy = pm25_clean + noise

        # Ensure non-negative (PM2.5 cannot be negative)
        pm25_noisy = np.maximum(pm25_noisy, 0)

        df = df.copy()
        df['pm2.5_clean'] = pm25_clean
        df['pm2.5_observed'] = pm25_noisy

        print(f"Added Gaussian noise: ζ={noise_level}, σ_X={pm25_std:.2f}, σ_noise={noise_level*pm25_std:.2f}")

        return df

    def create_irregular_observations(self,
                                     df: pd.DataFrame,
                                     observation_rate: float = 0.5,
                                     observation_dependent: bool = False,
                                     dependency_params: Optional[Dict] = None,
                                     seed: Optional[int] = None) -> pd.DataFrame:
        """
        Create irregular observation times by subsampling.

        Args:
            df: DataFrame with observations
            observation_rate: Base observation probability (if not dependent)
            observation_dependent: Whether observation probability depends on previous value
            dependency_params: Dict with 'alpha', 'beta', 'threshold', 'scale' for sigmoid
                              p_i = sigmoid(α + β·(O_{i-1} - τ)/s)
            seed: Random seed

        Returns:
            DataFrame with 'is_observed' column (boolean mask)
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(df)

        if not observation_dependent:
            # Simple random subsampling
            is_observed = np.random.rand(n) < observation_rate
            # Always observe first point
            is_observed[0] = True

        else:
            # Observation probability depends on previous observed value
            if dependency_params is None:
                # Default: higher PM2.5 → higher observation probability
                pm25_75percentile = df['pm2.5_observed'].quantile(0.75)
                dependency_params = {
                    'alpha': -1.0,  # Base log-odds
                    'beta': 2.0,    # Sensitivity to high values
                    'threshold': pm25_75percentile,
                    'scale': df['pm2.5_observed'].std()
                }

            alpha = dependency_params['alpha']
            beta = dependency_params['beta']
            threshold = dependency_params['threshold']
            scale = dependency_params['scale']

            is_observed = np.zeros(n, dtype=bool)
            is_observed[0] = True  # Always observe first

            for i in range(1, n):
                # Find previous observed value
                prev_obs_idx = np.where(is_observed[:i])[0]
                if len(prev_obs_idx) > 0:
                    last_observed_val = df['pm2.5_observed'].iloc[prev_obs_idx[-1]]
                else:
                    last_observed_val = df['pm2.5_observed'].iloc[0]

                # Compute observation probability
                z = alpha + beta * (last_observed_val - threshold) / scale
                prob = sigmoid(z)
                is_observed[i] = np.random.rand() < prob

        df = df.copy()
        df['is_observed'] = is_observed

        obs_rate_actual = is_observed.mean()
        print(f"Observation rate: {obs_rate_actual:.1%} ({is_observed.sum()}/{n})")

        if observation_dependent:
            # Show correlation between observation probability and PM2.5 level
            high_pm25 = df['pm2.5_observed'] > df['pm2.5_observed'].quantile(0.75)
            obs_rate_high = df.loc[high_pm25, 'is_observed'].mean()
            obs_rate_low = df.loc[~high_pm25, 'is_observed'].mean()
            print(f"  High PM2.5 (>75%ile): {obs_rate_high:.1%} observed")
            print(f"  Low PM2.5 (≤75%ile): {obs_rate_low:.1%} observed")

        return df

    def create_path_batches(self,
                           df: pd.DataFrame,
                           path_length_hours: int = 168,  # 1 week
                           stride_hours: int = 24,        # 1 day overlap
                           include_features: List[str] = None) -> Dict:
        """
        Convert DataFrame to path-based format for NJODE training.

        Args:
            df: Preprocessed DataFrame with 'is_observed' mask
            path_length_hours: Length of each path in hours
            stride_hours: Stride between consecutive paths
            include_features: List of feature columns to include (besides PM2.5)

        Returns:
            Dictionary with NJODE-compatible format:
            {
                'times': observation times,
                'time_ptr': batch pointers,
                'X': observed values (with features),
                'X_clean': clean PM2.5 values (for evaluation),
                'obs_idx': batch index for each observation,
                'start_X': initial values,
                'M': true latent process (for evaluation)
            }
        """
        if 'is_observed' not in df.columns:
            raise ValueError("Must call create_irregular_observations() first")

        # Default features: meteorological variables
        if include_features is None:
            feature_cols = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
            feature_cols = [c for c in feature_cols if c in df.columns]
            # Add wind direction dummies
            wind_cols = [c for c in df.columns if c.startswith('wind_')]
            feature_cols.extend(wind_cols)
        else:
            feature_cols = include_features

        # Create paths by sliding window
        n_hours = len(df)
        n_paths = (n_hours - path_length_hours) // stride_hours + 1

        all_times = []
        all_observations = []
        all_clean = []
        all_features = []
        time_ptr = [0]
        obs_idx = []
        start_X_list = []
        M_list = []

        actual_path_id = 0  # Track actual number of valid paths created
        for path_id in range(n_paths):
            start_idx = path_id * stride_hours
            end_idx = start_idx + path_length_hours

            if end_idx > n_hours:
                break

            path_df = df.iloc[start_idx:end_idx]

            # Extract observed timepoints
            obs_mask = path_df['is_observed'].values
            obs_indices = np.where(obs_mask)[0]

            if len(obs_indices) < 2:
                # Skip paths with too few observations
                continue

            # Times (normalize to [0, 1] within each path)
            times_hours = obs_indices.astype(float)
            times_normalized = times_hours / path_length_hours

            # Observations (PM2.5 + features)
            pm25_obs = path_df['pm2.5_observed'].values[obs_mask]
            features = path_df[feature_cols].values[obs_mask]

            # Normalize features (z-score per feature)
            features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)

            # Combine PM2.5 and features
            observations = np.concatenate([pm25_obs[:, None], features], axis=1)

            # Clean PM2.5 (ground truth)
            pm25_clean = path_df['pm2.5_clean'].values[obs_mask]

            # Store
            all_times.append(times_normalized)
            all_observations.append(observations)
            all_clean.append(pm25_clean)
            time_ptr.append(time_ptr[-1] + len(obs_indices))
            obs_idx.extend([actual_path_id] * len(obs_indices))  # Use actual_path_id, not path_id
            actual_path_id += 1  # Increment only for valid paths

            # Initial observation
            start_X_list.append(observations[0])

            # True latent process (all timepoints, for evaluation)
            M_path = path_df['pm2.5_clean'].values
            M_list.append(M_path)

        # Concatenate all paths
        dataset = {
            'times': np.concatenate(all_times),
            'time_ptr': time_ptr,
            'X': np.concatenate(all_observations, axis=0),
            'X_clean': np.concatenate(all_clean),
            'obs_idx': obs_idx,
            'start_X': np.array(start_X_list),
            'M': M_list,  # List of arrays (different lengths)
            'n_paths': len(start_X_list),
            'dimension': observations.shape[1],
            'feature_names': ['pm2.5'] + feature_cols
        }

        print(f"Created {dataset['n_paths']} paths with {len(dataset['times'])} total observations")
        print(f"Dimension: {dataset['dimension']} (PM2.5 + {len(feature_cols)} features)")
        print(f"Avg observations per path: {len(dataset['times']) / dataset['n_paths']:.1f}")

        return dataset

    def save_dataset(self, dataset: Dict, save_path: str):
        """Save dataset to pickle file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved dataset to {save_path}")


def create_beijing_pm25_datasets(
    data_path: str = "../data/PRSA_data_2010.1.1-2014.12.31.csv",
    output_dir: str = "../data/",
    noise_levels: List[float] = [0.25, 0.5, 1.0],
    observation_rates: List[float] = [0.3, 0.5, 0.7],
    use_observation_dependency: bool = True,
    seed: int = 42
) -> Dict[str, str]:
    """
    Complete pipeline to create multiple Beijing PM2.5 datasets with varying noise and observation rates.

    Args:
        data_path: Path to raw CSV
        output_dir: Output directory for processed datasets
        noise_levels: List of noise coefficients ζ
        observation_rates: List of base observation rates
        use_observation_dependency: Whether to use state-dependent observation
        seed: Random seed

    Returns:
        Dictionary mapping dataset names to file paths
    """
    loader = BeijingPM25Dataset(data_path)

    # Load and preprocess
    print("Loading and preprocessing Beijing PM2.5 data...")
    loader.load_and_preprocess()

    # Split train/val/test
    print("\nSplitting train/val/test...")
    train_df, val_df, test_df = loader.train_val_test_split()

    dataset_paths = {}

    # Create datasets for different noise levels and observation rates
    for noise_level in noise_levels:
        for obs_rate in observation_rates:
            # Create dataset name
            obs_type = "obsdep" if use_observation_dependency else "uniform"
            dataset_name = f"BeijingPM25_noise{noise_level}_obs{obs_rate}_{obs_type}"

            print(f"\n{'='*60}")
            print(f"Creating dataset: {dataset_name}")
            print(f"{'='*60}")

            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                print(f"\nProcessing {split_name} split...")

                # Add noise
                df_noisy = loader.add_observation_noise(split_df, noise_level=noise_level, seed=seed)

                # Create irregular observations
                df_irregular = loader.create_irregular_observations(
                    df_noisy,
                    observation_rate=obs_rate,
                    observation_dependent=use_observation_dependency,
                    seed=seed + hash(split_name) % 1000  # Different seed per split
                )

                # Create path batches
                dataset = loader.create_path_batches(
                    df_irregular,
                    path_length_hours=168,  # 1 week
                    stride_hours=24         # 1 day stride
                )

                # Save
                save_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.pkl")
                loader.save_dataset(dataset, save_path)
                dataset_paths[f"{dataset_name}_{split_name}"] = save_path

    print(f"\n{'='*60}")
    print(f"Created {len(dataset_paths)} datasets")
    print(f"{'='*60}")

    return dataset_paths


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Create Beijing PM2.5 datasets for NJODE")
    parser.add_argument("--data_path", type=str,
                       default="../data/PRSA_data_2010.1.1-2014.12.31.csv",
                       help="Path to raw Beijing PM2.5 CSV")
    parser.add_argument("--output_dir", type=str, default="../data/",
                       help="Output directory for processed datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create datasets
    dataset_paths = create_beijing_pm25_datasets(
        data_path=args.data_path,
        output_dir=args.output_dir,
        noise_levels=[0.25, 0.5, 1.0],
        observation_rates=[0.5],  # Use single rate for simplicity
        use_observation_dependency=True,
        seed=args.seed
    )

    print("\nDataset creation complete!")
    print("Dataset paths:")
    for name, path in dataset_paths.items():
        print(f"  {name}: {path}")
