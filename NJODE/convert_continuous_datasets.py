"""
Convert continuous Beijing PM2.5 datasets from per-path to per-time format.
"""

import pickle
import numpy as np
from pathlib import Path


def convert_dataset(input_path):
    """Convert dataset from per-path to per-time format."""
    print(f"\nConverting: {input_path}")

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    times_old = data['times']
    time_ptr_old = data['time_ptr']
    X_old = data['X']
    obs_idx_old = data['obs_idx']

    print(f"  Old format: times={len(times_old)}, time_ptr={len(time_ptr_old)}, unique_times={len(np.unique(times_old))}")

    # Check if already converted
    if len(times_old) + 1 == len(time_ptr_old):
        print(f"  Already in per-time format, skipping")
        return False

    # Build observations list
    observations = []
    for i in range(len(X_old)):
        path_idx = obs_idx_old[i]
        t = times_old[i]
        x = X_old[i]
        x_clean = data['X_clean'][i]
        observations.append({
            'time': t,
            'path_idx': path_idx,
            'X': x,
            'X_clean': x_clean
        })

    # Sort by (time, path_idx)
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
            # New time point
            if current_time is not None:
                time_ptr_new.append(len(X_new))
            times_new.append(obs['time'])
            current_time = obs['time']

        X_new.append(obs['X'])
        X_clean_new.append(obs['X_clean'])
        obs_idx_new.append(obs['path_idx'])

    # Final pointer
    time_ptr_new.append(len(X_new))

    # Update dataset
    data['times'] = np.array(times_new)
    data['time_ptr'] = time_ptr_new
    data['X'] = np.array(X_new)
    data['X_clean'] = np.array(X_clean_new)
    data['obs_idx'] = obs_idx_new

    # Save
    with open(input_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  New format: times={len(times_new)}, time_ptr={len(time_ptr_new)}")
    print(f"  ✓ Converted successfully")

    return True


def main():
    """Convert all continuous datasets."""

    data_dir = Path("../data/")

    # Find all continuous datasets
    pattern = "BeijingPM25_continuous_*.pkl"
    files = list(data_dir.glob(pattern))

    print(f"Found {len(files)} continuous datasets to check")

    converted = 0
    skipped = 0

    for file_path in sorted(files):
        if convert_dataset(file_path):
            converted += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already converted): {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
