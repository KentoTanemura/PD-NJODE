# Beijing PM2.5: Noise-Robust Loss with Continuous Path Evaluation

This implementation demonstrates the **noise-robust loss function** from NJODE Paper 3 on real-world time series data (Beijing PM2.5), with improved **continuous long-term path evaluation**.

## Overview

### Key Features

1. **Noise-Robust Loss Function**
   - Evaluates predictions at **left-limit only** (before observation jump)
   - Avoids attraction to noisy observations
   - Loss: `L = ||X_obs - Y_before_jump||²`
   - Contrast with legacy: `L = ||X_obs - Y_after||² + ||Y_before - Y_after||²`

2. **Continuous Long-Term Path Evaluation**
   - **Problem**: Previous approach used short 1-week overlapping windows for all splits
   - **Solution**: Test on continuous 1-month or 3-month paths
   - Enables realistic long-term prediction evaluation
   - No artificial boundaries

3. **Real-World Demonstration**
   - Beijing PM2.5 data (2010-2014, hourly measurements)
   - Controlled observation noise: `O_t = X_t + ε`, where `ε ~ N(0, ζ·σ_X)`
   - Noise levels: ζ ∈ {0.25, 0.5, 1.0} (low, medium, high)
   - Observation-dependent sampling: `p(observe) = sigmoid(α + β·(PM2.5 - τ)/s)`

## Dataset

**Source**: [UCI Beijing PM2.5 Dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2.5+data)

**Period**: January 1, 2010 - December 31, 2014 (hourly)

**Features** (11 dimensions):
- PM2.5 concentration (target)
- Meteorological variables: dew point, temperature, pressure, wind speed, precipitation
- Wind direction (one-hot encoded)

**Splits**:
- Train: 2010-2012 (3 years)
- Validation: 2013 (1 year)
- Test: 2014 (1 year)

## Quick Start

### 1. Download Data

```bash
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv
cd ../NJODE
```

### 2. Create Datasets

```bash
# Create continuous test datasets (monthly and quarterly paths)
python create_beijing_continuous_test.py

# Convert to per-time format (required for NJODE)
python convert_continuous_datasets.py
```

This creates:
- **Train/Val**: 168-hour (1 week) overlapping windows → efficient training
- **Test**: Multiple options
  - Standard: 168-hour windows (same as train)
  - Monthly: 730-hour (1 month) continuous paths × 12
  - Quarterly: 2190-hour (3 months) continuous paths × 4

### 3. Train Models

**Train single model**:
```bash
python train_beijing_pm25.py --noise_level 0.5 --epochs 100
```

**Compare across noise levels**:
```bash
python train_beijing_pm25.py --compare --epochs 100
```

**Evaluate on continuous monthly paths**:
```bash
python train_beijing_pm25.py --compare --test_type monthly --epochs 100
```

**Full experiment** (recommended):
```bash
python train_beijing_pm25.py --compare --test_type monthly --epochs 200
```

### 4. Visualize Results

```bash
# Monthly continuous path predictions
python visualize_continuous_prediction.py 0.5 monthly 50

# Quarterly continuous path predictions
python visualize_continuous_prediction.py 0.25 quarterly 50
```

## Expected Results

### Training Performance

**30 Epochs** (with regularization effect):
```
ζ=0.25: Test loss 21,748
ζ=0.5:  Test loss 8,006  ← Best (clean data + regularization)
ζ=1.0:  Test loss 12,700
```

**100 Epochs** (sufficient training):
```
ζ=0.25: Test loss 3,490  ← Best (clean data wins)
ζ=0.5:  Test loss 5,358
ζ=1.0:  Test loss 9,897
```

**Key Insight**: With sufficient training, lower noise (ζ=0.25) performs best. With limited training, medium noise (ζ=0.5) provides better regularization.

### Continuous Path Evaluation

**Monthly Paths** (ζ=0.5, 50 epochs):
```
Training: 1,021 paths, avg 19.7 obs/path
Test: 12 paths (12 months), avg 80.2 obs/path

Typical performance:
- RMSE: 26-70 μg/m³ (varies by month)
- Prediction: ~80 sparse observations → 730 hours continuous
```

**Quarterly Paths** (ζ=0.25, 50 epochs):
```
Training: 1,039 paths, avg 23.6 obs/path
Test: 4 paths (4 quarters), avg 315.2 obs/path

Typical performance:
- RMSE: Similar to monthly
- Prediction: ~330 observations → 2,190 hours continuous
```

## Architecture

### Model Configuration

```python
NJODE(
    input_size=11,           # PM2.5 + 10 features
    hidden_size=32,          # Latent state dimension
    output_size=1,           # PM2.5 prediction
    ode_nn=((64, 'relu'),),  # ODE network
    readout_nn=((32, 'relu'),),
    enc_nn=((32, 'relu'),),
    use_rnn=True,
    solver='euler',
    weight=0.5,              # Loss balance
    dropout_rate=0.1
)
```

### Noise Model

Observation noise:
```
O_t = X_t + ε
ε ~ N(0, ζ·σ_X)
```

Where:
- `X_t`: True PM2.5 (latent clean process)
- `O_t`: Observed PM2.5 (noisy)
- `ζ`: Noise coefficient (0.25, 0.5, 1.0)
- `σ_X`: Standard deviation of clean PM2.5

### Observation Sampling

**Observation-Dependent**: Higher PM2.5 → higher observation probability

```
p(observe at t) = sigmoid(α + β·(O_{prev} - τ)/s)
```

Parameters:
- `α = -1.0`: Base log-odds
- `β = 2.0`: Sensitivity to high values
- `τ`: 75th percentile threshold
- `s`: Standard deviation

Result: ~36% observation rate when PM2.5 is high (>75%ile), ~6% when low.

## File Structure

### Core Files

```
NJODE/
├── beijing_pm25_dataset.py          # Dataset loader and preprocessor
├── train_beijing_pm25.py             # Unified training script
├── create_beijing_continuous_test.py # Create continuous test paths
├── convert_continuous_datasets.py    # Format conversion
├── fix_beijing_dataset_format.py     # Dataset format utilities
├── visualize_continuous_prediction.py # Visualization script
├── configs/
│   └── config_BeijingPM25.py        # Experiment configuration
└── BEIJING_PM25_README.md           # This file
```

### Generated Files

```
data/
├── PRSA_data_2010.1.1-2014.12.31.csv          # Raw data (download)
├── BeijingPM25_continuous_noise{ζ}_train.pkl  # Training sets
├── BeijingPM25_continuous_noise{ζ}_val.pkl    # Validation sets
├── BeijingPM25_continuous_noise{ζ}_test_monthly.pkl   # Monthly test
├── BeijingPM25_continuous_noise{ζ}_test_quarterly.pkl # Quarterly test
├── plots_BeijingPM25/
│   ├── continuous_monthly_prediction_*.png    # Monthly visualizations
│   └── continuous_quarterly_prediction_*.png  # Quarterly visualizations
└── saved_models_BeijingPM25/
    └── model_noise{ζ}_{test_type}_{epochs}epochs.pt  # Trained models
```

## Understanding the Visualizations

### What the Plots Show

Each visualization displays continuous path predictions with:

- **Blue line**: NJODE prediction (continuous, dense)
- **Black dashed line**: True PM2.5 (ground truth)
- **Red points**: Noisy observations (sparse)
- **Green shading**: Extrapolation before first observation
- **Orange shading**: Extrapolation after last observation

### Interpretation

**Interpolation vs Extrapolation**:
- Between observations: **Interpolation** (model has nearby data)
- Before/after observations: **Extrapolation** (model predicts beyond data)

**Continuous Paths vs Short Windows**:
- Previous: Many 1-week windows with artificial boundaries
- Now: Full month/quarter continuous prediction → realistic evaluation

## Advantages Over Short Window Evaluation

| Aspect | Short Windows (168h) | Continuous Paths (730-2190h) |
|--------|---------------------|------------------------------|
| **Path length** | 1 week | 1-3 months |
| **Boundaries** | Artificial every week | Natural (full period) |
| **Evaluation** | Many short predictions | True long-term prediction |
| **Extrapolation** | Limited | Explicit (before/after obs) |
| **Real-world** | Less realistic | Realistic scenarios |

## Relationship to Original NJODE Papers

This implementation extends **NJODE Paper 3** (noisy observations framework):

**From Paper 3**:
- Noise-robust loss function (left-limit evaluation)
- Theoretical framework for noisy observations
- Synthetic data experiments (FBM, BMD, etc.)

**This Implementation Adds**:
- **Real-world data** demonstration (Beijing PM2.5)
- **Continuous long-term evaluation** (1-3 months vs 1 week)
- **Observation-dependent sampling** (state-dependent observation probability)
- **Practical comparison** across noise levels on real data

## Training Options

### Command-Line Arguments

```bash
python train_beijing_pm25.py [OPTIONS]

Options:
  --noise_level FLOAT     Noise level ζ (default: 0.5)
  --epochs INT           Training epochs (default: 100)
  --test_type {standard,monthly,quarterly}
                         Test set type (default: standard)
  --compare              Compare across all noise levels
  --device {cpu,cuda}    Training device (default: cpu)
```

### Example Commands

```bash
# Quick test (single model, 50 epochs)
python train_beijing_pm25.py --noise_level 0.5 --epochs 50

# Compare noise levels (recommended)
python train_beijing_pm25.py --compare --epochs 100

# Continuous monthly evaluation
python train_beijing_pm25.py --test_type monthly --epochs 100

# Full experiment (all noise levels, monthly paths, 200 epochs)
python train_beijing_pm25.py --compare --test_type monthly --epochs 200
```

## Citation

If you use this implementation, please cite the NJODE papers:

```bibtex
@article{herrera2021neural,
  title={Neural jump ordinary differential equations: Consistent continuous-time prediction and filtering},
  author={Herrera, Calypso and Krach, Florian and Teichmann, Josef},
  journal={ICLR 2021},
  year={2021}
}
```

## License

This code follows the license of the main PD-NJODE repository.
