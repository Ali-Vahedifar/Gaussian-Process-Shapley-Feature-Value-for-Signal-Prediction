# Quick Start Guide

This guide will help you get started with the GP+SFV framework in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction.git
cd Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Train a Model (5 minutes)

The simplest way to get started is to train on synthetic data:

```bash
python train.py --architecture resnet --epochs 50 --n_features 5
```

This will:
- Generate synthetic haptic data
- Train a GP oracle
- Compute Shapley feature values
- Train a ResNet with selected features
- Save the trained model to `checkpoints/best_model.pth`

**Expected output:**
```
Using device: cuda
STAGE 1a: Training Gaussian Process Oracle
GP Log Marginal Likelihood: -245.3421
STAGE 1b: Computing Shapley Feature Values
Selected 5 features: [0, 1, 2, 5, 8]
STAGE 2: Training Neural Network with GP Guidance
Epoch 50/50:
  Train Loss: 0.012345
  Val Loss: 0.013456
Training completed successfully!
Final validation loss: 0.013456
```

### 2. Evaluate the Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

**Expected output:**
```
Overall Metrics:
  Accuracy: 94.23%
  MSE: 0.003456
  RMSE: 0.058786
  MAE: 0.042134
  R² Score: 0.9234

Plots saved to evaluation_results/
```

### 3. Use Trained Model for Prediction

```python
import torch
import numpy as np
from models import create_model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')

# Recreate model
model = create_model(
    architecture='resnet',
    input_dim=len(checkpoint['selected_features']),
    output_dim=9
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
X_new = torch.randn(1, 9)  # New haptic signal
X_selected = X_new[:, checkpoint['selected_features']]

with torch.no_grad():
    prediction = model(X_selected)

print(f"Predicted haptic signal: {prediction}")
```

## Understanding the Framework

### Three Main Components

1. **Gaussian Process Oracle** (`gaussian_process.py`)
   - Learns probability distribution of haptic signals
   - Provides uncertainty quantification
   - Acts as "teacher" for neural network

2. **Shapley Feature Values** (`shapley_feature_value.py`)
   - Identifies most important features
   - Reduces computation by 72%
   - Based on cooperative game theory

3. **Neural Network** (`models.py`)
   - Fast real-time inference (2.2ms per sample)
   - Learns from GP oracle using JSD loss
   - Three architectures: FC, LSTM, ResNet

### Training Pipeline

```
Historical Data → GP Oracle → Shapley Values → Feature Selection
                                                      ↓
                                              Neural Network
                                                      ↓
                                              Fast Predictions
```

## Next Steps

### Work with Real Data

1. Prepare your dataset in the correct format:
   ```python
   # Shape: (n_timesteps, 9) where 9 features are:
   # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force_x, force_y, force_z]
   data = np.load('your_haptic_data.npy')
   np.save('data/my_dataset.npy', data)
   ```

2. Train on your data:
   ```bash
   python train.py --dataset my_dataset --epochs 100
   ```

### Experiment with Hyperparameters

```bash
# Try different architectures
python train.py --architecture lstm --epochs 100

# Adjust learning rate
python train.py --lr 0.001 --epochs 100

# Select different number of features
python train.py --n_features 7 --epochs 100
```

### Compare Methods

Train multiple models to compare:

```bash
# Baseline (no feature selection)
python train.py --n_features 9 --epochs 100

# With Shapley selection (5 features)
python train.py --n_features 5 --epochs 100

# With Shapley selection (3 features)
python train.py --n_features 3 --epochs 100
```

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 16

# Or use CPU
python train.py --device cpu
```

### Slow Shapley Computation

The Shapley value computation can be slow for large feature sets. For faster results:

```python
# In shapley_feature_value.py, reduce samples:
shapley_values = sfv.compute_shapley_values(
    n_samples=50,  # Reduce from default 100
    epochs_per_subset=20  # Reduce from default 30
)
```

### ImportError

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Getting Help

- **Documentation**: See [README.md](README.md) for detailed documentation
- **Issues**: Report bugs at [GitHub Issues](https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction/issues)
- **Email**: Contact av@ece.au.dk or qz@ece.au.dk

## Example Notebooks

Coming soon! Jupyter notebooks with step-by-step tutorials will be added to the `examples/` directory.

## Performance Tips

1. **Use GPU**: Training is 10x faster on GPU
2. **Batch size**: Larger batches (32-64) train faster but use more memory
3. **Feature selection**: Using 5-7 features gives best accuracy/speed tradeoff
4. **Early stopping**: Enable with `--patience 20` to avoid overfitting

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{vahedifar2026shapley,
  title={Shapley Features for Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Arthur and Zhang, Qi},
  booktitle={ICASSP 2026},
  year={2026}
}
```

---

**Ready to dive deeper?** Check out the full [README.md](README.md) for advanced usage and detailed API documentation.
