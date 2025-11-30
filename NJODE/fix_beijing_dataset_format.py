"""
Fix Beijing PM2.5 dataset format to match NJODE expectations.

The current format has time_ptr per-path, but NJODE expects per-time.
This script converts the format.
"""

import pickle
import numpy as np
from pathlib import Path

def convert_dataset_format(input_path, output_path):
    """
    Convert from per-path to per-time format.

    Current format:
    - times: [t_00, t_01, ..., t_10, t_11, ...] (one time per observation, sorted by path)
    - time_ptr[i]: start index for path i's observations
    - obs_idx[j]: which path observation j belongs to

    Required format:
    - times: [unique_t_0, unique_t_1, ...] (sorted unique times)
    - time_ptr[i]: start index for all observations at times[i]
    - obs_idx[j]: which path observation j belongs to
    """
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Original format:")
    print(f"  times shape: {data['times'].shape}")
    print(f"  time_ptr length: {len(data['time_ptr'])}")
    print(f"  Unique times: {len(np.unique(data['times']))}")
    print(f"  n_paths: {data['n_paths']}")

    # Get current data
    times_old = data['times']
    time_ptr_old = data['time_ptr']
    X_old = data['X']
    X_clean_old = data['X_clean']
    obs_idx_old = data['obs_idx']

    # Create mapping from (time, path) -> observation_data
    observations = []
    for path_idx in range(data['n_paths']):
        start = time_ptr_old[path_idx]
        end = time_ptr_old[path_idx + 1]

        for obs_idx_in_path in range(start, end):
            t = times_old[obs_idx_in_path]
            observations.append({
                'time': t,
                'path_idx': path_idx,
                'X': X_old[obs_idx_in_path],
                'X_clean': X_clean_old[obs_idx_in_path],
            })

    # Sort by time first, then by path_idx for determinism
    observations.sort(key=lambda x: (x['time'], x['path_idx']))

    # Build new format
    times_new = []
    time_ptr_new = [0]
    X_new = []
    X_clean_new = []
    obs_idx_new = []

    current_time = None
    for obs in observations:
        if obs['time'] != current_time:
            # New unique time
            if current_time is not None:
                time_ptr_new.append(len(X_new))
            times_new.append(obs['time'])
            current_time = obs['time']

        X_new.append(obs['X'])
        X_clean_new.append(obs['X_clean'])
        obs_idx_new.append(obs['path_idx'])

    # Final time_ptr entry
    time_ptr_new.append(len(X_new))

    # Convert to arrays
    times_new = np.array(times_new)
    X_new = np.array(X_new)
    X_clean_new = np.array(X_clean_new)

    print(f"\nNew format:")
    print(f"  times shape: {times_new.shape}")
    print(f"  time_ptr length: {len(time_ptr_new)}")
    print(f"  X shape: {X_new.shape}")
    print(f"  Verification: len(times) + 1 == len(time_ptr)? {len(times_new) + 1 == len(time_ptr_new)}")

    # Create new dataset
    data_new = {
        'times': times_new,
        'time_ptr': time_ptr_new,
        'X': X_new,
        'X_clean': X_clean_new,
        'obs_idx': obs_idx_new,
        'start_X': data['start_X'],
        'M': data['M'],
        'n_paths': data['n_paths'],
        'dimension': data['dimension'],
        'feature_names': data['feature_names'],
        'delta_t': 0.001,  # For ODE solver
        'maturity': 1.0,   # Normalized time
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data_new, f)

    print(f"✓ Done!")

    return data_new

if __name__ == "__main__":
    # Convert all Beijing PM2.5 datasets
    data_dir = Path("../data")

    datasets = [
        "BeijingPM25_noise0.25_obs0.5_obsdep_train.pkl",
        "BeijingPM25_noise0.25_obs0.5_obsdep_val.pkl",
        "BeijingPM25_noise0.25_obs0.5_obsdep_test.pkl",
        "BeijingPM25_noise0.5_obs0.5_obsdep_train.pkl",
        "BeijingPM25_noise0.5_obs0.5_obsdep_val.pkl",
        "BeijingPM25_noise0.5_obs0.5_obsdep_test.pkl",
        "BeijingPM25_noise1.0_obs0.5_obsdep_train.pkl",
        "BeijingPM25_noise1.0_obs0.5_obsdep_val.pkl",
        "BeijingPM25_noise1.0_obs0.5_obsdep_test.pkl",
    ]

    for dataset_name in datasets:
        input_path = data_dir / dataset_name
        if not input_path.exists():
            print(f"Skipping {dataset_name} (not found)")
            continue

        # Overwrite original file
        output_path = input_path

        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")

        convert_dataset_format(input_path, output_path)

    print(f"\n{'='*70}")
    print(f"All datasets converted successfully!")
    print(f"{'='*70}")
