"""
Beijing PM2.5 Training Script - Unified Interface

This script provides a unified interface for training NJODE models on Beijing PM2.5 data
with noise-robust loss functions and continuous path evaluation.

Usage:
    # Train single model
    python train_beijing_pm25.py --noise_level 0.5 --epochs 100

    # Compare across noise levels
    python train_beijing_pm25.py --compare --epochs 100

    # Use continuous test paths
    python train_beijing_pm25.py --test_type monthly --epochs 100

    # Full experiment
    python train_beijing_pm25.py --compare --test_type monthly --epochs 200
"""

import sys
import argparse
import pickle
import torch
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

from models import NJODE
from loss_functions import compute_loss_noise_robust


def load_dataset(dataset_path: str) -> Dict:
    """Load dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_model(
    input_size: int,
    hidden_size: int = 32,
    output_size: int = 1,
    device: str = 'cpu'
) -> NJODE:
    """
    Create NJODE model for PM2.5 prediction.

    Args:
        input_size: Input dimension (11 for PM2.5 + 10 features)
        hidden_size: Hidden state dimension
        output_size: Output dimension (1 for PM2.5)
        device: Device for training

    Returns:
        NJODE model
    """
    model = NJODE(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        ode_nn=((64, 'relu'),),
        readout_nn=((32, 'relu'),),
        enc_nn=((32, 'relu'),),
        use_rnn=True,
        bias=True,
        dropout_rate=0.1,
        solver='euler',
        weight=0.5,
        weight_decay=1.0,
        input_coords=list(range(input_size)),
        output_coords=[0],
        signature_coords=[],
        options={'which_loss': 'noise_robust', 'input_sig': False, 'level': 2},
    ).to(device)

    return model


def train_epoch(
    model: NJODE,
    data: Dict,
    optimizer: optim.Optimizer,
    device: str = 'cpu'
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()

    times = data['times']
    time_ptr = data['time_ptr']
    X = torch.tensor(data['X'], dtype=torch.float32, device=device)
    obs_idx = torch.tensor(data['obs_idx'], dtype=torch.long, device=device)
    start_X = torch.tensor(data['start_X'], dtype=torch.float32, device=device)

    # Compute n_obs_ot from obs_idx
    n_obs_ot = torch.zeros(data['n_paths'], dtype=torch.float32, device=device)
    for idx in obs_idx:
        n_obs_ot[idx] += 1

    delta_t = float(data.get('delta_t', 0.001))
    T = float(data.get('maturity', 1.0))

    # Forward pass
    hT, loss = model(
        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
        delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
        return_path=False, get_loss=True, M=None, start_M=None
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(
    model: NJODE,
    data: Dict,
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    """
    Evaluate model on dataset.

    Returns:
        Tuple of (loss, metrics_dict)
    """
    model.eval()

    with torch.no_grad():
        times = data['times']
        time_ptr = data['time_ptr']
        X = torch.tensor(data['X'], dtype=torch.float32, device=device)
        obs_idx = torch.tensor(data['obs_idx'], dtype=torch.long, device=device)
        start_X = torch.tensor(data['start_X'], dtype=torch.float32, device=device)

        n_obs_ot = torch.zeros(data['n_paths'], dtype=torch.float32, device=device)
        for idx in obs_idx:
            n_obs_ot[idx] += 1

        delta_t = float(data.get('delta_t', 0.001))
        T = float(data.get('maturity', 1.0))

        hT, loss = model(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
            return_path=False, get_loss=True, M=None, start_M=None
        )

        # Compute additional metrics if X_clean is available
        metrics = {'loss': loss.item()}

        if 'X_clean' in data:
            X_clean = data['X_clean']
            X_pred = X[:, 0].cpu().numpy()  # First dimension is PM2.5

            if X_clean.ndim > 1:
                X_clean = X_clean[:, 0]

            mse = np.mean((X_pred - X_clean) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(X_pred - X_clean))

            metrics.update({
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            })

    return loss.item(), metrics


def train_model(
    noise_level: float,
    n_epochs: int = 100,
    test_type: str = 'standard',
    device: str = 'cpu',
    save_model: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Train a single model configuration.

    Args:
        noise_level: Noise level ζ (0.25, 0.5, 1.0)
        n_epochs: Number of training epochs
        test_type: Type of test set ('standard', 'monthly', 'quarterly')
        device: Device for training
        save_model: Whether to save the trained model
        verbose: Whether to print progress

    Returns:
        Dictionary with training history and final metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"  Noise level: ζ={noise_level}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Test type: {test_type}")
        print(f"  Device: {device}")
        print(f"{'='*70}")

    # Load datasets
    data_train = load_dataset(f'../data/BeijingPM25_continuous_noise{noise_level}_train.pkl')
    data_val = load_dataset(f'../data/BeijingPM25_continuous_noise{noise_level}_val.pkl')

    if test_type == 'standard':
        data_test = load_dataset(f'../data/BeijingPM25_continuous_noise{noise_level}_test.pkl')
    elif test_type == 'monthly':
        data_test = load_dataset(f'../data/BeijingPM25_continuous_noise{noise_level}_test_monthly.pkl')
    elif test_type == 'quarterly':
        data_test = load_dataset(f'../data/BeijingPM25_continuous_noise{noise_level}_test_quarterly.pkl')
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    if verbose:
        print(f"\nDataset sizes:")
        print(f"  Train: {data_train['n_paths']} paths, {len(data_train['X'])} observations")
        print(f"  Val: {data_val['n_paths']} paths, {len(data_val['X'])} observations")
        print(f"  Test: {data_test['n_paths']} paths, {len(data_test['X'])} observations")

    # Create model
    model = create_model(
        input_size=data_train['dimension'],
        hidden_size=32,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': []
    }

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, data_train, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, _ = evaluate(model, data_val, device)
        history['val_loss'].append(val_loss)

        # Test
        test_loss, test_metrics = evaluate(model, data_test, device)
        history['test_loss'].append(test_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Train={train_loss:.1f}, Val={val_loss:.1f}, Test={test_loss:.1f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    # Final evaluation
    final_test_loss, final_metrics = evaluate(model, data_test, device)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"  Best val loss: {best_val_loss:.1f} (epoch {best_epoch+1})")
        print(f"  Final test loss: {final_test_loss:.1f}")
        if 'rmse' in final_metrics:
            print(f"  Test RMSE: {final_metrics['rmse']:.1f} μg/m³")
            print(f"  Test MAE: {final_metrics['mae']:.1f} μg/m³")
        print(f"{'='*70}")

    # Save model
    if save_model:
        save_dir = Path(f'../data/saved_models_BeijingPM25/')
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f'model_noise{noise_level}_{test_type}_{n_epochs}epochs.pt'
        torch.save(model.state_dict(), model_path)
        if verbose:
            print(f"\nModel saved to: {model_path}")

    return {
        'noise_level': noise_level,
        'n_epochs': n_epochs,
        'test_type': test_type,
        'history': history,
        'final_metrics': final_metrics,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }


def compare_noise_levels(
    n_epochs: int = 100,
    test_type: str = 'standard',
    device: str = 'cpu'
) -> Dict:
    """
    Compare training across different noise levels.

    Args:
        n_epochs: Number of epochs per configuration
        test_type: Test set type
        device: Device for training

    Returns:
        Dictionary with comparison results
    """
    noise_levels = [0.25, 0.5, 1.0]
    results = {}

    print(f"\n{'='*70}")
    print(f"Comparing Noise Levels: ζ ∈ {noise_levels}")
    print(f"Epochs: {n_epochs}, Test type: {test_type}")
    print(f"{'='*70}")

    for noise_level in noise_levels:
        result = train_model(
            noise_level=noise_level,
            n_epochs=n_epochs,
            test_type=test_type,
            device=device,
            save_model=True,
            verbose=True
        )
        results[f'zeta_{noise_level}'] = result

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Noise ζ':<10} {'Test Loss':<12} {'RMSE':<12} {'MAE':<12}")
    print(f"{'-'*70}")

    for noise_level in noise_levels:
        key = f'zeta_{noise_level}'
        metrics = results[key]['final_metrics']
        test_loss = metrics['loss']
        rmse = metrics.get('rmse', float('nan'))
        mae = metrics.get('mae', float('nan'))
        print(f"{noise_level:<10.2f} {test_loss:<12.1f} {rmse:<12.1f} {mae:<12.1f}")

    print(f"{'='*70}")

    # Save results
    results_dir = Path('../data/results_BeijingPM25')
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f'comparison_{test_type}_{n_epochs}epochs.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train NJODE on Beijing PM2.5 data with noise-robust loss'
    )

    parser.add_argument('--noise_level', type=float, default=0.5,
                       help='Noise level ζ (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--test_type', type=str, default='standard',
                       choices=['standard', 'monthly', 'quarterly'],
                       help='Test set type (default: standard)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare across all noise levels')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for training (default: cpu)')

    args = parser.parse_args()

    if args.compare:
        # Compare across noise levels
        compare_noise_levels(
            n_epochs=args.epochs,
            test_type=args.test_type,
            device=args.device
        )
    else:
        # Train single configuration
        train_model(
            noise_level=args.noise_level,
            n_epochs=args.epochs,
            test_type=args.test_type,
            device=args.device,
            save_model=True,
            verbose=True
        )


if __name__ == "__main__":
    main()
