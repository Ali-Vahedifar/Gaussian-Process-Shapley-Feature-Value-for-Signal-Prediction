"""
Gaussian Process Module for Tactile Internet Signal Prediction
================================================================
This module implements Gaussian Process regression for haptic signal prediction
in Tactile Internet applications, serving as an oracle for neural network training.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
Authors: Mohammad Ali Vahedifar, Arthur, and Qi Zhang
"""

import torch
import numpy as np
from typing import Tuple, Optional
import warnings


class GaussianProcess:
    """
    Gaussian Process Regression Model for haptic signal prediction.
    
    The GP provides probabilistic predictions and serves as ground truth
    oracle for training neural networks in the TI framework.
    
    Attributes:
        kernel_type: Type of kernel function ('rbf', 'matern', or 'linear')
        length_scale: Length scale parameter for the kernel
        sigma_f: Signal variance parameter
        sigma_y: Noise variance parameter
        device: Computation device (cpu or cuda)
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        sigma_f: float = 1.0,
        sigma_y: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Gaussian Process model.
        
        Args:
            kernel_type: Type of kernel ('rbf', 'matern', or 'linear')
            length_scale: Controls smoothness of predictions
            sigma_f: Signal variance, controls output scale
            sigma_y: Noise variance, accounts for observation noise
            device: Computing device for tensor operations
        """
        # Store hyperparameters for kernel computation
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        
        # Set computation device (GPU if available for faster inference)
        self.device = device
        
        # Initialize storage for training data (will be set during fit)
        self.X_train = None  # Historical input signals
        self.Y_train = None  # Corresponding target signals
        self.K_inv = None    # Inverse of kernel matrix (precomputed for efficiency)
        
    def kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel (covariance) matrix between two sets of inputs.
        
        The kernel function k(x, x') measures the similarity between inputs,
        determining how correlated their outputs should be.
        
        Args:
            X1: First set of inputs, shape (n1, d)
            X2: Second set of inputs, shape (n2, d)
            
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        if self.kernel_type == 'rbf':
            # Radial Basis Function (RBF) kernel: k(x,x') = σ²_f * exp(-||x-x'||²/(2l²))
            # This is the most commonly used kernel, providing smooth predictions
            
            # Compute squared Euclidean distance between all pairs
            # Shape after unsqueeze: X1 (n1, 1, d), X2 (1, n2, d)
            # Broadcasting gives distance matrix of shape (n1, n2, d)
            dist = torch.sum((X1.unsqueeze(1) - X2.unsqueeze(0)) ** 2, dim=2)
            
            # Apply RBF kernel formula with length_scale and signal variance
            return self.sigma_f ** 2 * torch.exp(-dist / (2 * self.length_scale ** 2))
            
        elif self.kernel_type == 'matern':
            # Matérn kernel with ν=3/2: k(x,x') = σ²_f * (1 + √3*r/l) * exp(-√3*r/l)
            # where r = ||x-x'||. Less smooth than RBF, suitable for rougher functions
            
            # Compute Euclidean distance (not squared)
            dist = torch.sqrt(torch.sum((X1.unsqueeze(1) - X2.unsqueeze(0)) ** 2, dim=2) + 1e-8)
            
            # Apply Matérn 3/2 kernel formula
            sqrt3_dist = np.sqrt(3) * dist / self.length_scale
            return self.sigma_f ** 2 * (1 + sqrt3_dist) * torch.exp(-sqrt3_dist)
            
        elif self.kernel_type == 'linear':
            # Linear kernel: k(x,x') = σ²_f * x^T x'
            # Simplest kernel, assumes linear relationships between inputs and outputs
            
            # Compute dot product between all pairs: X1 @ X2^T
            return self.sigma_f ** 2 * torch.mm(X1, X2.t())
            
        else:
            # Raise error for unsupported kernel types
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> None:
        """
        Fit the Gaussian Process to training data.
        
        This precomputes the kernel matrix and its inverse for efficient prediction.
        In the paper, this represents the offline training phase where GP learns
        the distribution of haptic signals.
        
        Args:
            X_train: Training inputs of shape (n, d) where n is number of samples
            Y_train: Training targets of shape (n, output_dim)
        """
        # Move training data to the specified device (GPU/CPU)
        self.X_train = X_train.to(self.device)
        self.Y_train = Y_train.to(self.device)
        
        # Compute kernel matrix K(X, X) for all training points
        # This captures the covariance structure of the training data
        K = self.kernel(self.X_train, self.X_train)
        
        # Add noise variance to diagonal: K + σ²_y * I
        # This accounts for observation noise and ensures numerical stability
        K_noisy = K + self.sigma_y ** 2 * torch.eye(K.shape[0], device=self.device)
        
        # Precompute inverse for prediction efficiency
        # We use Cholesky decomposition for numerical stability
        try:
            # Cholesky decomposition: K = L * L^T where L is lower triangular
            L = torch.linalg.cholesky(K_noisy)
            
            # Solve for K^{-1} using the Cholesky factor
            # This is more stable than directly computing the inverse
            self.K_inv = torch.cholesky_inverse(L)
            
        except RuntimeError:
            # If Cholesky fails (matrix not positive definite), add regularization
            warnings.warn("Cholesky decomposition failed, adding regularization")
            
            # Add small regularization term to diagonal for numerical stability
            K_noisy += 1e-6 * torch.eye(K.shape[0], device=self.device)
            
            # Recompute Cholesky and inverse
            L = torch.linalg.cholesky(K_noisy)
            self.K_inv = torch.cholesky_inverse(L)
    
    def predict(
        self,
        X_test: torch.Tensor,
        return_std: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions for test inputs with uncertainty quantification.
        
        This implements the GP predictive distribution:
        μ* = K(X*, X)[K(X,X) + σ²I]^{-1}Y
        Σ* = K(X*,X*) - K(X*,X)[K(X,X) + σ²I]^{-1}K(X,X*)
        
        Args:
            X_test: Test inputs of shape (n_test, d)
            return_std: Whether to return predictive standard deviation
            
        Returns:
            mean: Predictive mean of shape (n_test, output_dim)
            std: Predictive standard deviation (if return_std=True)
        """
        # Move test data to device
        X_test = X_test.to(self.device)
        
        # Compute kernel between test and training points: K(X*, X)
        # This measures how test points relate to training points
        K_star = self.kernel(X_test, self.X_train)
        
        # Compute predictive mean: μ* = K(X*, X) * K^{-1} * Y
        # This is the expected value of the prediction
        mean = torch.mm(torch.mm(K_star, self.K_inv), self.Y_train)
        
        if return_std:
            # Compute kernel of test points with themselves: K(X*, X*)
            K_star_star = self.kernel(X_test, X_test)
            
            # Compute predictive covariance: Σ* = K(X*,X*) - K(X*,X) K^{-1} K(X,X*)
            # This quantifies prediction uncertainty
            cov = K_star_star - torch.mm(torch.mm(K_star, self.K_inv), K_star.t())
            
            # Extract diagonal elements (variances) and take square root for std dev
            # Diagonal elements represent variance of each prediction
            std = torch.sqrt(torch.diag(cov).unsqueeze(1).expand_as(mean) + 1e-8)
            
            return mean, std
        else:
            return mean, None
    
    def log_marginal_likelihood(self) -> float:
        """
        Compute the log marginal likelihood of the training data.
        
        This metric can be used for hyperparameter optimization via gradient descent.
        Higher values indicate better fit to the data.
        
        Returns:
            Log marginal likelihood value
        """
        # Compute kernel matrix K(X, X) with noise
        K = self.kernel(self.X_train, self.X_train)
        K_noisy = K + self.sigma_y ** 2 * torch.eye(K.shape[0], device=self.device)
        
        # Cholesky decomposition for efficient computation
        L = torch.linalg.cholesky(K_noisy)
        
        # Solve L * alpha = Y for alpha
        alpha = torch.cholesky_solve(self.Y_train, L)
        
        # Compute log marginal likelihood:
        # log p(y|X) = -0.5 * y^T * K^{-1} * y - 0.5 * log|K| - (n/2) * log(2π)
        
        # First term: -0.5 * y^T * K^{-1} * y (data fit term)
        fit_term = -0.5 * torch.sum(self.Y_train * alpha)
        
        # Second term: -0.5 * log|K| = -sum(log(diag(L))) (complexity penalty)
        complexity_term = -torch.sum(torch.log(torch.diag(L)))
        
        # Third term: -(n/2) * log(2π) (constant normalization)
        n = self.X_train.shape[0]
        constant_term = -0.5 * n * np.log(2 * np.pi)
        
        # Sum all terms for total log marginal likelihood
        return (fit_term + complexity_term + constant_term).item()
    
    def update_hyperparameters(
        self,
        length_scale: Optional[float] = None,
        sigma_f: Optional[float] = None,
        sigma_y: Optional[float] = None
    ) -> None:
        """
        Update GP hyperparameters and refit the model.
        
        This allows for online adaptation of the GP as new data arrives.
        
        Args:
            length_scale: New length scale (if provided)
            sigma_f: New signal variance (if provided)
            sigma_y: New noise variance (if provided)
        """
        # Update provided hyperparameters
        if length_scale is not None:
            self.length_scale = length_scale
        if sigma_f is not None:
            self.sigma_f = sigma_f
        if sigma_y is not None:
            self.sigma_y = sigma_y
        
        # Refit the model with new hyperparameters if training data exists
        if self.X_train is not None and self.Y_train is not None:
            self.fit(self.X_train, self.Y_train)


def create_gp_oracle(
    X_history: np.ndarray,
    Y_history: np.ndarray,
    kernel_type: str = 'rbf',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> GaussianProcess:
    """
    Factory function to create and fit a GP oracle for TI applications.
    
    This function encapsulates the GP creation process described in the paper,
    where the GP serves as a probabilistic oracle for generating ground truth
    estimates that guide neural network training.
    
    Args:
        X_history: Historical haptic signal inputs, shape (n_samples, n_features)
        Y_history: Corresponding future signals, shape (n_samples, output_dim)
        kernel_type: Type of kernel function to use
        device: Computation device
        
    Returns:
        Fitted GaussianProcess object ready for prediction
    """
    # Convert numpy arrays to PyTorch tensors for GPU acceleration
    X_train = torch.from_numpy(X_history).float()
    Y_train = torch.from_numpy(Y_history).float()
    
    # Initialize GP with default hyperparameters
    # These can be optimized using cross-validation or marginal likelihood
    gp = GaussianProcess(
        kernel_type=kernel_type,
        length_scale=1.0,     # Controls smoothness; smaller = more variable
        sigma_f=1.0,          # Signal variance; scales the output
        sigma_y=0.1,          # Noise level; higher = less trust in observations
        device=device
    )
    
    # Fit the GP to historical data (offline training phase)
    gp.fit(X_train, Y_train)
    
    return gp


if __name__ == "__main__":
    """
    Example usage demonstrating GP oracle for haptic signal prediction.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic haptic data (force, velocity, position over time)
    n_samples = 100
    n_features = 9  # 3 DoF × 3 measurements (force, velocity, position)
    
    # Create synthetic training data simulating haptic signals
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    Y_train = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create and fit GP oracle
    print("Creating GP oracle...")
    gp = create_gp_oracle(X_train, Y_train, kernel_type='rbf')
    
    # Generate test data for prediction
    X_test = np.random.randn(10, n_features).astype(np.float32)
    X_test_tensor = torch.from_numpy(X_test).float()
    
    # Make predictions with uncertainty quantification
    print("Making predictions...")
    mean, std = gp.predict(X_test_tensor, return_std=True)
    
    # Display results
    print(f"\nPredictive mean shape: {mean.shape}")
    print(f"Predictive std shape: {std.shape}")
    print(f"\nSample predictions (first 3 features):")
    print(f"Mean: {mean[0, :3].cpu().numpy()}")
    print(f"Std:  {std[0, :3].cpu().numpy()}")
    
    # Compute log marginal likelihood (for hyperparameter tuning)
    log_ml = gp.log_marginal_likelihood()
    print(f"\nLog marginal likelihood: {log_ml:.4f}")
    
    print("\nGP oracle created successfully!")
