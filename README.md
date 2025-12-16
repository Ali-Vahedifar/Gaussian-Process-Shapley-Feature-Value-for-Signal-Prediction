# Shapley Features for Robust Signal Prediction in Tactile Internet

Official PyTorch implementation of the paper:

**"Shapley Features for Robust Signal Prediction in Tactile Internet"**  
*Mohammad Ali Vahedifar, Arthur, and Qi Zhang*  
ICASSP 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository implements a novel two-stage predictive framework for Tactile Internet (TI) applications that combines:

1. **Gaussian Process (GP) Oracle**: Provides probabilistic ground-truth estimates for haptic signal prediction
2. **Shapley Feature Values (SFV)**: Identifies the most informative features for prediction
3. **Neural Networks**: Fast inference with distributional alignment via Jensen-Shannon Divergence loss

### Key Results

- **95.72% accuracy** on ResNet+GP+SFV (11.1% improvement over LeFo baseline)
- **72% inference time reduction** compared to GP alone
- **27% speedup** when SFV is applied to the LeFo method

<p align="center">
  <img src="https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction/blob/main/SFV.drawio.png" alt="Framework Overview" width="800"/>
</p>


## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Ali-Vahedifar/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction.git
cd Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

The paper evaluates on 7 real-world haptic datasets:

| Dataset | Description |
|---------|-------------|
| D1 | Drag Max Stiffness Y |
| D2 | Horizontal Movement Fast |
| D3 | Horizontal Movement Slow |
| D4 | Tap and Hold Max Stiffness Z-Fast |
| D5 | Tap and Hold Max Stiffness Z-Slow |
| D6 | Tapping Max Stiffness Y-Z |
| D7 | Tapping Max Stiffness Z |

Datasets are recorded using a Novint Falcon haptic device in a Chai3D virtual environment, capturing 3D position, velocity, and force measurements.

### Dataset Format

Each dataset should be a `.npy` file with shape `(n_timesteps, 9)` where the 9 features are:
```
[position_x, position_y, position_z,
 velocity_x, velocity_y, velocity_z,
 force_x, force_y, force_z]
```

Place datasets in the `data/` directory.

## 🎯 Quick Start

### Basic Training

Train a ResNet model with GP+SFV on synthetic data:

```bash
python train.py --architecture resnet --epochs 100 --n_features 5
```

### Training on Real Dataset

```bash
python train.py \
    --dataset drag_max_stiffness_y \
    --architecture resnet \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.01 \
    --n_features 5
```

### Advanced Usage

```bash
python train.py \
    --dataset drag_max_stiffness_y \
    --architecture resnet \
    --epochs 200 \
    --batch_size 64 \
    --lr 0.01 \
    --n_features 7 \
    --device cuda
```

## 📖 Usage Examples

### 1. Train Gaussian Process Oracle

```python
from gaussian_process import create_gp_oracle
import numpy as np

# Load your haptic data
X_history = np.load('data/X_history.npy')  # Shape: (n_samples, n_features)
Y_history = np.load('data/Y_history.npy')  # Shape: (n_samples, output_dim)

# Create and train GP oracle
gp = create_gp_oracle(
    X_history=X_history,
    Y_history=Y_history,
    kernel_type='rbf',
    device='cuda'
)

# Make predictions with uncertainty
X_test = np.load('data/X_test.npy')
mean, std = gp.predict(torch.from_numpy(X_test).float(), return_std=True)

print(f"Predicted mean: {mean}")
print(f"Prediction uncertainty: {std}")
```

### 2. Compute Shapley Feature Values

```python
from shapley_feature_value import select_features_with_shapley
from models import create_model

# Create a simple model for evaluation
model = create_model('fc', input_dim=9, output_dim=9)

# Select top 5 features using Shapley values
selected_features, importance = select_features_with_shapley(
    model=model,
    X_train=X_train,
    Y_train=Y_train,
    k=5  # Select top 5 features
)

print(f"Selected features: {selected_features}")
print(f"Feature importance: {importance}")
```

### 3. Train Neural Network with JSD Loss

```python
from models import create_model
from loss_functions import create_loss_function
import torch.optim as optim

# Create ResNet model (best performing in paper)
model = create_model('resnet', input_dim=5, output_dim=9)

# Create JSD loss function
criterion = create_loss_function('jsd')

# Setup optimizer (SGD with momentum as in paper)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        # Get GP predictions (ground truth distribution)
        gp_mean, gp_std = gp.predict(X_batch, return_std=True)
        
        # Neural network predictions
        nn_predictions = model(X_batch[:, selected_features])
        nn_std = torch.ones_like(nn_predictions) * 0.1
        
        # Compute JSD loss
        loss, jsd, mse = criterion(nn_predictions, nn_std, gp_mean, gp_std)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 📁 Project Structure

```
shapley-gp-ti/
├── gaussian_process.py        # GP oracle implementation
├── shapley_feature_value.py   # SFV computation
├── models.py                   # Neural network architectures (FC, LSTM, ResNet)
├── loss_functions.py           # JSD loss and variants
├── train.py                    # Main training script
├── data_utils.py              # Data loading and preprocessing
├── evaluate.py                # Model evaluation (to be created)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # License file
└── data/                      # Dataset directory
    ├── drag_max_stiffness_y.npy
    ├── horizontal_movement_fast.npy
    └── ...
```

## 🧪 Reproducing Paper Results

To reproduce the results from Table 2 (Drag Max Stiffness Y Dataset):

### 1. Fully Connected Network

```bash
python train.py --dataset drag_max_stiffness_y --architecture fc --n_features 5
```

### 2. LSTM Network

```bash
python train.py --dataset drag_max_stiffness_y --architecture lstm --n_features 5
```

### 3. ResNet (Best Results)

```bash
python train.py --dataset drag_max_stiffness_y --architecture resnet --n_features 5
```

## 📈 Results

### Accuracy Comparison (Drag Max Stiffness Y Dataset)

| Architecture | LeFo | LeFo+SFV | **GP+SFV** |
|-------------|------|----------|------------|
| FC | 82.57% | 88.36% (+7.0%) | **92.59%** (+12.1%) |
| LSTM | 83.50% | 86.84% (+4.0%) | **91.71%** (+9.8%) |
| ResNet | 86.19% | 90.82% (+5.4%) | **95.72%** (+11.1%) |

### Inference Time Comparison (ms per sample)

| Method | LeFo | LeFo+SFV | GP | **GP+SFV** | Speedup |
|--------|------|----------|-----|------------|---------|
| Average | 12.4 | 8.8 | 7.3 | **2.2** | **72%** |

## 🔬 Technical Details

### Gaussian Process

- **Kernel**: RBF (Radial Basis Function)
- **Hyperparameters**: 
  - Length scale: 1.0
  - Signal variance (σ_f): 1.0
  - Noise variance (σ_y): 0.1

### Shapley Feature Values

- **Computation**: Monte Carlo approximation for >10 features
- **Samples**: 100 subset evaluations per feature
- **Metric**: Mean Squared Error (MSE)

### Neural Networks

#### Fully Connected (FC)
- 12 layers × 100 units
- ReLU activation
- Dropout: 0.1

#### LSTM
- 2 stacked layers × 128 units
- Dropout: 0.1 between layers
- Dense output layer

#### ResNet-32
- 8 residual blocks
- Hidden dimension: 256
- He initialization
- Dropout: 0.1

### Training Configuration

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01 (initial)
- **Batch Size**: 32
- **Loss Function**: Jensen-Shannon Divergence + MSE
- **Early Stopping**: Patience = 20 epochs

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{vahedifar2026shapley,
  title={Shapley Features for Robust Signal Prediction in Tactile Internet},
  author={Vahedifar, Mohammad Ali and Arthur and Zhang, Qi},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was supported by:
- TOAST project (EU Horizon Europe, Grant No. 101073465)
- Danish Council for Independent Research project eTouch (Grant No. 1127-00339B)
- NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043)

## 📧 Contact

- Ali Vahedi - av@ece.au.dk

**DIGIT and Department of Electrical and Computer Engineering**  
Aarhus University, Denmark

## 🔗 Links

- [Paper](https://arxiv.org/abs/2509.21032) 
- [TOAST Project](https://toast-doctoral-network.eu/)
- [Aarhus University ECE](https://ece.au.dk/)

---

**Note**: If you find any bugs or have suggestions for improvements, please open an issue!
