"""
Jensen-Shannon Divergence Loss Module
======================================
This module implements the Jensen-Shannon Divergence (JSD) loss function
used to train neural networks to match the GP oracle's probability distributions.

JSD is a symmetric measure of similarity between two probability distributions,
ensuring that predicted haptic signals statistically match the GP's estimates.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class JSDLoss(nn.Module):
    """
    Jensen-Shannon Divergence Loss Function.
    
    The JSD loss measures the statistical similarity between the neural network's
    predicted distribution (Q_NN) and the GP oracle's distribution (P_GP).
    
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q) is the average distribution.
    
    Properties:
    - Symmetric: JSD(P||Q) = JSD(Q||P)
    - Bounded: 0 ≤ JSD ≤ log(2)
    - Zero iff P = Q (distributions are identical)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        epsilon: float = 1e-10
    ):
        """
        Initialize JSD Loss.
        
        Args:
            reduction: How to reduce batch losses ('mean', 'sum', or 'none')
            epsilon: Small constant for numerical stability
        """
        super(JSDLoss, self).__init__()
        
        # Store configuration
        self.reduction = reduction
        self.epsilon = epsilon  # Prevent log(0) errors
    
    def kl_divergence(
        self,
        P: torch.Tensor,
        Q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Kullback-Leibler divergence between distributions P and Q.
        
        KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
        
        Args:
            P: First probability distribution, shape (batch_size, n_classes)
            Q: Second probability distribution, shape (batch_size, n_classes)
            
        Returns:
            KL divergence value(s), shape depends on reduction
        """
        # Add epsilon to prevent log(0) and division by zero
        P_safe = P + self.epsilon
        Q_safe = Q + self.epsilon
        
        # Compute KL divergence: Σ P * log(P/Q)
        # log(P/Q) = log(P) - log(Q) for numerical stability
        kl = torch.sum(P_safe * (torch.log(P_safe) - torch.log(Q_safe)), dim=-1)
        
        return kl
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence between predicted and target distributions.
        
        For Gaussian distributions, we use a closed-form approximation based on
        discretizing the distributions into probability bins.
        
        Args:
            pred_mean: Predicted means from NN, shape (batch_size, output_dim)
            pred_std: Predicted standard deviations, shape (batch_size, output_dim)
            target_mean: Target means from GP, shape (batch_size, output_dim)
            target_std: Target standard deviations, shape (batch_size, output_dim)
            
        Returns:
            JSD loss value (scalar if reduction='mean'/'sum')
        """
        # Create discretized probability distributions from Gaussian parameters
        # We sample points around the means and compute probability densities
        
        # Define range for discretization (cover ±3 standard deviations)
        n_bins = 50  # Number of discretization points
        
        # Determine range for discretization (union of both distributions' ranges)
        # Cover ±3 sigma around both means
        min_val = torch.min(
            pred_mean - 3 * pred_std,
            target_mean - 3 * target_std
        )
        max_val = torch.max(
            pred_mean + 3 * pred_std,
            target_mean + 3 * target_std
        )
        
        # Create discretization points (bins)
        # Shape: (batch_size, output_dim, n_bins)
        bins = torch.linspace(0, 1, n_bins, device=pred_mean.device)
        bins = bins.view(1, 1, -1)  # Reshape for broadcasting
        bins = min_val.unsqueeze(-1) + bins * (max_val - min_val).unsqueeze(-1)
        
        # Compute Gaussian probability density at each bin
        # For predicted distribution Q_NN
        pred_probs = self._gaussian_pdf(bins, pred_mean.unsqueeze(-1), pred_std.unsqueeze(-1))
        
        # For target distribution P_GP
        target_probs = self._gaussian_pdf(bins, target_mean.unsqueeze(-1), target_std.unsqueeze(-1))
        
        # Normalize to ensure probabilities sum to 1
        pred_probs = pred_probs / (torch.sum(pred_probs, dim=-1, keepdim=True) + self.epsilon)
        target_probs = target_probs / (torch.sum(target_probs, dim=-1, keepdim=True) + self.epsilon)
        
        # Compute average distribution M = 0.5 * (P + Q)
        M = 0.5 * (pred_probs + target_probs)
        
        # Compute JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        # Reshape for KL computation: (batch_size * output_dim, n_bins)
        batch_size, output_dim = pred_mean.shape
        P_flat = target_probs.reshape(-1, n_bins)
        Q_flat = pred_probs.reshape(-1, n_bins)
        M_flat = M.reshape(-1, n_bins)
        
        # Compute both KL divergences
        kl_pm = self.kl_divergence(P_flat, M_flat)
        kl_qm = self.kl_divergence(Q_flat, M_flat)
        
        # Combine KL divergences for JSD
        jsd = 0.5 * (kl_pm + kl_qm)
        
        # Reshape back to (batch_size, output_dim)
        jsd = jsd.reshape(batch_size, output_dim)
        
        # Average over output dimensions
        jsd = torch.mean(jsd, dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(jsd)
        elif self.reduction == 'sum':
            return torch.sum(jsd)
        else:
            return jsd
    
    def _gaussian_pdf(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian probability density function.
        
        PDF(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
        
        Args:
            x: Points to evaluate, shape (..., n_bins)
            mean: Mean of Gaussian, shape (..., 1)
            std: Standard deviation, shape (..., 1)
            
        Returns:
            Probability densities at x, shape (..., n_bins)
        """
        # Compute standardized distance: (x - μ) / σ
        z = (x - mean) / (std + self.epsilon)
        
        # Compute Gaussian PDF
        # Normalization constant: 1 / (σ√(2π))
        norm_const = 1.0 / (std * np.sqrt(2 * np.pi) + self.epsilon)
        
        # Exponential term: exp(-0.5 * z²)
        exp_term = torch.exp(-0.5 * z ** 2)
        
        # Complete PDF
        pdf = norm_const * exp_term
        
        return pdf


class CombinedLoss(nn.Module):
    """
    Combined loss function: JSD Loss + MSE Loss.
    
    In practice, combining JSD with MSE can improve training stability.
    MSE ensures point predictions are accurate, while JSD ensures
    distributional alignment with the GP oracle.
    """
    
    def __init__(
        self,
        jsd_weight: float = 1.0,
        mse_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Initialize combined loss.
        
        Args:
            jsd_weight: Weight for JSD loss term
            mse_weight: Weight for MSE loss term
            reduction: Reduction method for both losses
        """
        super(CombinedLoss, self).__init__()
        
        # Store weights for loss components
        self.jsd_weight = jsd_weight
        self.mse_weight = mse_weight
        
        # Initialize loss functions
        self.jsd_loss = JSDLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_mean: Predicted means from NN
            pred_std: Predicted standard deviations from NN
            target_mean: Target means from GP
            target_std: Target standard deviations from GP
            
        Returns:
            total_loss: Weighted combination of JSD and MSE
            jsd_component: JSD loss value (for logging)
            mse_component: MSE loss value (for logging)
        """
        # Compute JSD loss (distributional similarity)
        jsd = self.jsd_loss(pred_mean, pred_std, target_mean, target_std)
        
        # Compute MSE loss (point prediction accuracy)
        mse = self.mse_loss(pred_mean, target_mean)
        
        # Combine losses with weights
        total = self.jsd_weight * jsd + self.mse_weight * mse
        
        return total, jsd, mse


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    
    Alternative to JSD that directly maximizes the likelihood of target
    distribution under the predicted Gaussian distribution.
    
    NLL = 0.5 * log(2π * σ²) + 0.5 * ((y - μ) / σ)²
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        epsilon: float = 1e-6
    ):
        """
        Initialize Gaussian NLL Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
            epsilon: Small constant for numerical stability
        """
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.
        
        Args:
            pred_mean: Predicted means, shape (batch_size, output_dim)
            pred_std: Predicted standard deviations, shape (batch_size, output_dim)
            target: Target values, shape (batch_size, output_dim)
            
        Returns:
            NLL loss value
        """
        # Ensure std is positive
        pred_std = pred_std + self.epsilon
        
        # Compute variance
        var = pred_std ** 2
        
        # Compute NLL = 0.5 * log(2π * σ²) + 0.5 * ((y - μ) / σ)²
        # First term: log normalization constant
        log_term = 0.5 * torch.log(2 * np.pi * var)
        
        # Second term: squared error normalized by variance
        sq_error_term = 0.5 * ((target - pred_mean) ** 2) / var
        
        # Total NLL
        nll = log_term + sq_error_term
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll


def create_loss_function(
    loss_type: str = 'jsd',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('jsd', 'combined', 'nll', or 'mse')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Initialized loss function
    """
    if loss_type == 'jsd':
        # Pure JSD loss (as described in paper)
        return JSDLoss(**kwargs)
        
    elif loss_type == 'combined':
        # JSD + MSE combination (for stability)
        return CombinedLoss(**kwargs)
        
    elif loss_type == 'nll':
        # Gaussian negative log-likelihood
        return GaussianNLLLoss(**kwargs)
        
    elif loss_type == 'mse':
        # Standard MSE (baseline)
        return nn.MSELoss(**kwargs)
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    """
    Test the loss functions.
    """
    # Set random seed
    torch.manual_seed(42)
    
    # Create sample data
    batch_size = 32
    output_dim = 9
    
    # Predicted distribution from neural network
    pred_mean = torch.randn(batch_size, output_dim)
    pred_std = torch.abs(torch.randn(batch_size, output_dim)) + 0.1
    
    # Target distribution from GP oracle
    target_mean = torch.randn(batch_size, output_dim)
    target_std = torch.abs(torch.randn(batch_size, output_dim)) + 0.1
    
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Test JSD Loss
    print("\n1. Jensen-Shannon Divergence Loss:")
    jsd_loss = JSDLoss()
    jsd_value = jsd_loss(pred_mean, pred_std, target_mean, target_std)
    print(f"   JSD Loss: {jsd_value.item():.6f}")
    
    # Test Combined Loss
    print("\n2. Combined Loss (JSD + MSE):")
    combined_loss = CombinedLoss(jsd_weight=1.0, mse_weight=0.5)
    total, jsd_comp, mse_comp = combined_loss(pred_mean, pred_std, target_mean, target_std)
    print(f"   Total Loss: {total.item():.6f}")
    print(f"   JSD Component: {jsd_comp.item():.6f}")
    print(f"   MSE Component: {mse_comp.item():.6f}")
    
    # Test Gaussian NLL Loss
    print("\n3. Gaussian Negative Log-Likelihood Loss:")
    nll_loss = GaussianNLLLoss()
    nll_value = nll_loss(pred_mean, pred_std, target_mean)
    print(f"   NLL Loss: {nll_value.item():.6f}")
    
    # Test standard MSE
    print("\n4. Standard MSE Loss:")
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(pred_mean, target_mean)
    print(f"   MSE Loss: {mse_value.item():.6f}")
    
    print("\n" + "=" * 60)
    print("All loss functions working correctly!")
