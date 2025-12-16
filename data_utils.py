"""
Data Utilities for Haptic Signal Processing
============================================
This module provides utilities for loading, preprocessing, and batching
haptic data from Tactile Internet experiments.

Supports the datasets from the paper:
- D1: Drag Max Stiffness Y
- D2: Horizontal Movement Fast
- D3: Horizontal Movement Slow
- D4: Tap and Hold Max Stiffness Z-Fast
- D5: Tap and Hold Max Stiffness Z-Slow
- D6: Tapping Max Stiffness Y-Z
- D7: Tapping Max Stiffness Z

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


class TactileDataset(Dataset):
    """
    PyTorch Dataset for haptic/tactile data.
    
    Handles 3D position, velocity, and force measurements from
    haptic devices in Tactile Internet applications.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize Tactile Dataset.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            Y: Target outputs, shape (n_samples, output_dim)
            transform: Optional transform to apply to samples
        """
        # Convert to float32 for PyTorch compatibility
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        
        # Store transform function
        self.transform = transform
        
        # Verify shapes match
        assert len(self.X) == len(self.Y), \
            f"X and Y must have same length, got {len(self.X)} and {len(self.Y)}"
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Tuple of (input_features, target_output)
        """
        # Get sample at index
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        
        # Apply transform if provided
        if self.transform:
            x_sample = self.transform(x_sample)
        
        return x_sample, y_sample


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 10,
    prediction_horizon: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time-series prediction.
    
    For predicting the next N samples based on previous M samples,
    this function creates input-output pairs using sliding windows.
    
    Args:
        data: Time-series data, shape (n_timesteps, n_features)
        window_size: Number of past timesteps to use as input
        prediction_horizon: Number of future timesteps to predict
        
    Returns:
        X: Input windows, shape (n_windows, window_size * n_features)
        Y: Target outputs, shape (n_windows, n_features)
    """
    n_timesteps, n_features = data.shape
    
    # Calculate number of valid windows
    n_windows = n_timesteps - window_size - prediction_horizon + 1
    
    if n_windows <= 0:
        raise ValueError(
            f"Not enough data: need at least {window_size + prediction_horizon} "
            f"timesteps, but got {n_timesteps}"
        )
    
    # Initialize arrays for inputs and outputs
    X = np.zeros((n_windows, window_size * n_features))
    Y = np.zeros((n_windows, n_features))
    
    # Create sliding windows
    for i in range(n_windows):
        # Input: flatten window of past observations
        # Shape: (window_size, n_features) -> (window_size * n_features,)
        X[i] = data[i:i + window_size].flatten()
        
        # Output: next sample after prediction_horizon
        # For simplicity, we predict sample at i + window_size + prediction_horizon - 1
        Y[i] = data[i + window_size + prediction_horizon - 1]
    
    return X, Y


def normalize_haptic_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize haptic data using training statistics.
    
    This ensures that all features are on similar scales, which
    improves neural network training stability.
    
    Args:
        X_train: Training data
        X_val: Validation data
        X_test: Test data
        method: Normalization method ('standard', 'minmax', or 'none')
        
    Returns:
        Normalized X_train, X_val, X_test, and fitted scaler
    """
    if method == 'none':
        # Return data unchanged
        return X_train, X_val, X_test, None
    
    elif method == 'standard':
        # Z-score normalization: (x - μ) / σ
        # Fit scaler on training data only (avoid data leakage)
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        
        # Apply same transformation to val and test
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm, scaler
    
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        # Scale to [0, 1] range
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm, scaler
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_haptic_dataset(
    dataset_name: str = 'synthetic',
    data_path: str = './data',
    window_size: int = 10,
    prediction_horizon: int = 10,
    val_split: float = 0.15,
    test_split: float = 0.15,
    normalize: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess haptic dataset.
    
    This function handles loading different datasets from the paper
    and prepares them for training. If the dataset file doesn't exist,
    it generates synthetic data for demonstration.
    
    Args:
        dataset_name: Name of dataset to load
        data_path: Path to data directory
        window_size: Size of sliding window for input
        prediction_horizon: Number of steps ahead to predict
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        normalize: Whether to normalize the data
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    import os
    
    # Try to load real dataset
    dataset_path = os.path.join(data_path, f"{dataset_name}.npy")
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        # Load numpy array with shape (n_timesteps, n_features)
        # Each feature is one of: position_x, position_y, position_z,
        #                        velocity_x, velocity_y, velocity_z,
        #                        force_x, force_y, force_z
        raw_data = np.load(dataset_path)
        
    else:
        # Generate synthetic haptic data for demonstration
        warnings.warn(
            f"Dataset {dataset_name} not found at {dataset_path}. "
            "Generating synthetic data for demonstration."
        )
        raw_data = generate_synthetic_haptic_data(
            n_timesteps=1000,
            random_state=random_state
        )
    
    print(f"Raw data shape: {raw_data.shape}")
    
    # Create sliding windows for time-series prediction
    print(f"Creating sliding windows (window_size={window_size}, "
          f"prediction_horizon={prediction_horizon})")
    X, Y = create_sliding_windows(
        data=raw_data,
        window_size=window_size,
        prediction_horizon=prediction_horizon
    )
    
    print(f"Windowed data: X={X.shape}, Y={Y.shape}")
    
    # Split into train, validation, and test sets
    # First split: separate test set
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y,
        test_size=test_split,
        random_state=random_state,
        shuffle=True  # Shuffle for better generalization
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_split / (1 - test_split)  # Adjust for already removed test set
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    # Normalize data if requested
    if normalize:
        print("Normalizing data...")
        X_train, X_val, X_test, scaler = normalize_haptic_data(
            X_train, X_val, X_test,
            method='standard'
        )
        
        # Also normalize targets (helps with training stability)
        Y_train, Y_val, Y_test, _ = normalize_haptic_data(
            Y_train, Y_val, Y_test,
            method='standard'
        )
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def generate_synthetic_haptic_data(
    n_timesteps: int = 1000,
    n_features: int = 9,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic haptic data for demonstration.
    
    Creates realistic-looking time-series data with correlations
    between position, velocity, and force measurements.
    
    Args:
        n_timesteps: Number of time steps to generate
        n_features: Number of features (default 9 = 3 DoF × 3 measurements)
        random_state: Random seed
        
    Returns:
        Synthetic haptic data, shape (n_timesteps, n_features)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Time array
    t = np.linspace(0, 10, n_timesteps)
    
    # Initialize data array
    data = np.zeros((n_timesteps, n_features))
    
    # Generate position signals (smooth trajectories)
    # Use sinusoidal patterns with different frequencies for each axis
    for i in range(3):  # 3 degrees of freedom
        # Position: smooth sinusoidal motion
        freq = 0.5 + i * 0.2  # Different frequency for each axis
        phase = i * np.pi / 3  # Phase offset
        data[:, i] = np.sin(2 * np.pi * freq * t + phase)
        
        # Velocity: derivative of position
        data[:, i + 3] = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
        
        # Force: proportional to velocity with damping
        # Simulates haptic interaction force
        data[:, i + 6] = -0.5 * data[:, i + 3] + 0.1 * np.random.randn(n_timesteps)
    
    # Add some noise to make it more realistic
    data += 0.05 * np.random.randn(n_timesteps, n_features)
    
    # Smooth the data slightly (haptic signals are typically smooth)
    from scipy.ndimage import gaussian_filter1d
    for i in range(n_features):
        data[:, i] = gaussian_filter1d(data[:, i], sigma=2)
    
    return data


def create_dataloaders(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TactileDataset(X_train, Y_train)
    val_dataset = TactileDataset(X_val, Y_val)
    test_dataset = TactileDataset(X_test, Y_test)
    
    # Create data loaders
    # Training loader: shuffle=True for better generalization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    # Validation loader: shuffle=False for consistent evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    # Test loader: shuffle=False for consistent evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Test data utilities.
    """
    print("Testing Data Utilities")
    print("=" * 60)
    
    # Test synthetic data generation
    print("\n1. Generating synthetic haptic data:")
    synthetic_data = generate_synthetic_haptic_data(n_timesteps=1000)
    print(f"   Generated data shape: {synthetic_data.shape}")
    print(f"   Data range: [{synthetic_data.min():.4f}, {synthetic_data.max():.4f}]")
    
    # Test sliding window creation
    print("\n2. Creating sliding windows:")
    X, Y = create_sliding_windows(synthetic_data, window_size=10, prediction_horizon=10)
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    
    # Test dataset loading
    print("\n3. Loading and preprocessing dataset:")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_haptic_dataset(
        dataset_name='synthetic',
        window_size=10,
        prediction_horizon=10
    )
    print(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Test DataLoader creation
    print("\n4. Creating DataLoaders:")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=32
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test batch loading
    print("\n5. Testing batch loading:")
    for X_batch, Y_batch in train_loader:
        print(f"   Batch X shape: {X_batch.shape}")
        print(f"   Batch Y shape: {Y_batch.shape}")
        break  # Just test first batch
    
    print("\n" + "=" * 60)
    print("All data utilities working correctly!")
