"""
Configuration file for Beijing PM2.5 experiments with noise-robust loss

This configuration demonstrates:
1. Real-world data with added observation noise
2. Noise-robust loss (left-limit only, no observation attraction)
3. Observation-dependent sampling
4. Comparison with legacy loss functions
"""

import numpy as np

# ==============================================================================
# Dataset Dictionaries
# ==============================================================================

# Base Beijing PM2.5 dataset parameters
BeijingPM25_base_dict = {
    'model_name': "BeijingPM25",
    'data_path': "../data/PRSA_data_2010.1.1-2014.12.31.csv",
    'path_length_hours': 168,  # 1 week paths
    'stride_hours': 24,        # 1 day overlap
    'train_years': [2010, 2011, 2012],
    'val_years': [2013],
    'test_years': [2014],
}

# Low noise (ζ=0.25), moderate observation rate
BeijingPM25_noise025_dict = {
    **BeijingPM25_base_dict,
    'dataset_id': 'noise025_obs05_obsdep',
    'noise_level': 0.25,
    'observation_rate': 0.5,
    'observation_dependent': True,
}

# Medium noise (ζ=0.5), moderate observation rate
BeijingPM25_noise050_dict = {
    **BeijingPM25_base_dict,
    'dataset_id': 'noise050_obs05_obsdep',
    'noise_level': 0.5,
    'observation_rate': 0.5,
    'observation_dependent': True,
}

# High noise (ζ=1.0), moderate observation rate
BeijingPM25_noise100_dict = {
    **BeijingPM25_base_dict,
    'dataset_id': 'noise100_obs05_obsdep',
    'noise_level': 1.0,
    'observation_rate': 0.5,
    'observation_dependent': True,
}

# ==============================================================================
# Model Parameter Lists
# ==============================================================================

# Baseline: Compare noise-robust loss vs legacy loss at different noise levels
param_list_BeijingPM25_baseline = {
    'epochs': [200],
    'batch_size': [50],
    'save_every': [10],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'hidden_size': [64],
    'bias': [True],
    'ode_nn': [((128, 'relu'), (128, 'relu'))],
    'readout_nn': [((64, 'relu'), (64, 'relu'))],
    'enc_nn': [((64, 'relu'),)],
    'use_rnn': [True],
    'dropout_rate': [0.1],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [False],
    'level': [2],
    'which_loss': ['noise_robust', 'legacy'],  # Compare both loss types
    'data_dict': ['BeijingPM25_noise025_dict', 'BeijingPM25_noise050_dict', 'BeijingPM25_noise100_dict'],
    'plot': [False],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': ['../data/saved_models_BeijingPM25/'],
}

# Extended: Test different model architectures with noise-robust loss
param_list_BeijingPM25_extended = {
    'epochs': [200],
    'batch_size': [50],
    'learning_rate': [0.001],
    'hidden_size': [32, 64, 128],  # Test different hidden sizes
    'ode_nn': [
        ((64, 'relu'), (64, 'relu')),
        ((128, 'relu'), (128, 'relu')),
        ((128, 'relu'), (128, 'relu'), (128, 'relu')),  # Deeper network
    ],
    'readout_nn': [((64, 'relu'), (64, 'relu'))],
    'enc_nn': [((64, 'relu'),)],
    'use_rnn': [True],
    'dropout_rate': [0.1],
    'solver': ["euler"],
    'weight': [0.5],
    'dataset': ['BeijingPM25_noise050_dict'],  # Focus on medium noise
    'loss_type': ['noise_robust'],
    'dataset_id': ['noise050_obs05_obsdep'],
}

# Ablation: Test importance of observation dependency
param_list_BeijingPM25_ablation = {
    'epochs': [200],
    'batch_size': [50],
    'learning_rate': [0.001],
    'hidden_size': [64],
    'ode_nn': [((128, 'relu'), (128, 'relu'))],
    'readout_nn': [((64, 'relu'), (64, 'relu'))],
    'enc_nn': [((64, 'relu'),)],
    'use_rnn': [True],
    'dropout_rate': [0.1],
    'solver': ["euler"],
    'weight': [0.5],
    'dataset': ['BeijingPM25_noise050_dict'],
    'loss_type': ['noise_robust'],
    'dataset_id': ['noise050_obs05_obsdep'],  # With observation dependency
    'observation_dependent': [True, False],    # Ablate dependency
}

# ==============================================================================
# Overview Dictionaries (for result summaries)
# ==============================================================================

overview_dict_BeijingPM25_baseline = {
    'model_ids': list(range(1, 7)),  # 3 noise levels × 2 loss types
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'which': 'test',
    'plot_loss': True,
}

overview_dict_BeijingPM25_extended = {
    'model_ids': list(range(1, 28)),  # 3 hidden_size × 3 ode_nn × 3 variations
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'which': 'test',
    'plot_loss': True,
}

# ==============================================================================
# Plot Configuration
# ==============================================================================

plot_paths_BeijingPM25_dict = {
    'model_ids': [1, 2, 5, 6],  # Compare noise-robust vs legacy at low and high noise
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'which': 'test',
    'paths_to_plot': [0, 1, 2, 3, 4],  # Plot first 5 test paths
    'plot_same_yaxis': False,
    'plot_errors': True,
    'save_path': '../data/plots_BeijingPM25/',
    'plot_obs_prob': True,  # Plot observation probability over time
}

# Detailed comparison plot for paper figures
plot_comparison_BeijingPM25_dict = {
    'model_ids': {
        'noise_025_robust': 1,
        'noise_025_legacy': 2,
        'noise_050_robust': 3,
        'noise_050_legacy': 4,
        'noise_100_robust': 5,
        'noise_100_legacy': 6,
    },
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'which': 'test',
    'paths_to_plot': [0, 5, 10],  # Selected interesting paths
    'time_window': [0.1, 0.3],    # Focus on specific time window (10-30% of path)
    'save_path': '../data/plots_BeijingPM25/comparison/',
    'figsize': (12, 8),
    'show_zoomed': True,  # Show zoomed-in view of interesting regions
}

# ==============================================================================
# Evaluation Configuration
# ==============================================================================

eval_dict_BeijingPM25 = {
    'model_ids': list(range(1, 7)),
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'metrics': [
        'mse_left_limit',      # MSE at observation left-limits (key metric)
        'mse_observations',    # MSE at observation points
        'mae_left_limit',      # MAE at observation left-limits
        'mae_observations',    # MAE at observation points
        'noise_sensitivity',   # How much predictions change immediately after observations
    ],
    'compute_confidence_intervals': True,
    'n_bootstrap': 1000,
    'save_path': '../data/evaluation_BeijingPM25/',
}

# ==============================================================================
# Cross-validation Configuration
# ==============================================================================

crossval_dict_BeijingPM25 = {
    'n_folds': 5,
    'val_size': 0.2,
    'dataset': 'BeijingPM25_noise050_dict',
    'params': param_list_BeijingPM25_baseline,
    'metric': 'mse_left_limit',
}

# ==============================================================================
# Noise Sensitivity Analysis
# ==============================================================================

# Test model robustness to varying noise at inference time
noise_sensitivity_dict_BeijingPM25 = {
    'model_ids': [1, 3, 5],  # Noise-robust models trained at different noise levels
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'test_noise_levels': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    'n_test_paths': 100,
    'save_path': '../data/noise_sensitivity_BeijingPM25/',
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_dataset_config(noise_level: float, obs_rate: float = 0.5, obs_dep: bool = True) -> dict:
    """
    Helper to generate dataset configuration for arbitrary noise level.

    Args:
        noise_level: Noise coefficient ζ
        obs_rate: Base observation rate
        obs_dep: Whether to use observation-dependent sampling

    Returns:
        Dataset configuration dictionary
    """
    dataset_id = f"noise{int(noise_level*100):03d}_obs{int(obs_rate*100):02d}_{'obsdep' if obs_dep else 'uniform'}"

    return {
        **BeijingPM25_base_dict,
        'dataset_id': dataset_id,
        'noise_level': noise_level,
        'observation_rate': obs_rate,
        'observation_dependent': obs_dep,
    }


def get_model_param_list(
    dataset_configs: list,
    loss_types: list = ['noise_robust', 'legacy'],
    hidden_sizes: list = [64],
    **kwargs
) -> dict:
    """
    Helper to generate model parameter list for multiple configurations.

    Args:
        dataset_configs: List of dataset configuration dicts
        loss_types: List of loss types to test
        hidden_sizes: List of hidden sizes to test
        **kwargs: Additional model parameters to override

    Returns:
        Parameter list dictionary
    """
    default_params = {
        'epochs': [200],
        'batch_size': [50],
        'learning_rate': [0.001],
        'hidden_size': hidden_sizes,
        'ode_nn': [((128, 'relu'), (128, 'relu'))],
        'readout_nn': [((64, 'relu'), (64, 'relu'))],
        'enc_nn': [((64, 'relu'),)],
        'use_rnn': [True],
        'dropout_rate': [0.1],
        'solver': ["euler"],
        'weight': [0.5],
        'dataset': [d['dataset_id'] + '_dict' for d in dataset_configs],
        'loss_type': loss_types,
    }

    # Override with user-provided params
    default_params.update(kwargs)

    return default_params


# Example: Quick experiment configuration
quick_test_dict = {
    'epochs': [50],  # Fewer epochs for quick testing
    'batch_size': [50],
    'save_every': [5],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'hidden_size': [32],  # Smaller model
    'bias': [True],
    'ode_nn': [((64, 'relu'),)],  # Single layer
    'readout_nn': [((32, 'relu'),)],
    'enc_nn': [((32, 'relu'),)],
    'use_rnn': [True],
    'dropout_rate': [0.1],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [False],
    'level': [2],
    'which_loss': ['noise_robust'],
    'data_dict': ['BeijingPM25_noise050_dict'],
    'plot': [False],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,)],
    'saved_models_path': ['../data/saved_models_BeijingPM25_test/'],
}

overview_dict_quick_test = {
    'model_ids': [1],
    'saved_models_path': '../data/saved_models_BeijingPM25/',
    'which': 'test',
}
