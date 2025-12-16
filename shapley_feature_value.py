"""
Shapley Feature Value Module for Feature Selection
===================================================
This module implements Shapley Feature Values (SFV) for identifying the most
informative features in Tactile Internet signal prediction.

Based on cooperative game theory, SFV provides a principled method for
feature attribution and selection.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Callable
from itertools import combinations, chain
from tqdm import tqdm
import warnings


class ShapleyFeatureValue:
    """
    Shapley Feature Value calculator for feature importance estimation.
    
    Implements the Shapley value from cooperative game theory to determine
    each feature's contribution to model performance. This satisfies key
    axioms: transferability, null contribution, symmetry, and linearity.
    
    The Shapley value for feature 'a' is:
    φ_a = Σ [|S|!(M-|S|-1)!/M!] * [E_{S∪{a}} - E_S]
    
    where S ranges over all subsets not containing 'a', and E represents
    the model's performance (explanation model).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        metric: str = 'mse',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Shapley Feature Value calculator.
        
        Args:
            model: PyTorch model to evaluate (the explanation model E)
            X_train: Training features of shape (n_samples, n_features)
            Y_train: Training targets of shape (n_samples, output_dim)
            metric: Performance metric ('mse', 'mae', or 'accuracy')
            device: Computation device
        """
        # Store model and data for feature evaluation
        self.model = model.to(device)
        self.X_train = X_train.to(device)
        self.Y_train = Y_train.to(device)
        self.device = device
        
        # Store number of features (M in the paper)
        self.n_features = X_train.shape[1]
        
        # Set performance metric for evaluating feature subsets
        self.metric = metric
        self.metric_fn = self._get_metric_function(metric)
        
        # Storage for computed Shapley values
        self.shapley_values = None
        
    def _get_metric_function(self, metric: str) -> Callable:
        """
        Get the appropriate metric function for model evaluation.
        
        Args:
            metric: Name of the metric ('mse', 'mae', 'accuracy')
            
        Returns:
            Function that computes the metric
        """
        if metric == 'mse':
            # Mean Squared Error: measures average squared prediction error
            def mse(y_true, y_pred):
                return torch.mean((y_true - y_pred) ** 2).item()
            return mse
            
        elif metric == 'mae':
            # Mean Absolute Error: measures average absolute prediction error
            def mae(y_true, y_pred):
                return torch.mean(torch.abs(y_true - y_pred)).item()
            return mae
            
        elif metric == 'accuracy':
            # Classification accuracy: fraction of correct predictions
            def accuracy(y_true, y_pred):
                # For regression, use threshold-based accuracy
                return torch.mean((torch.abs(y_true - y_pred) < 0.1).float()).item()
            return accuracy
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _train_subset_model(
        self,
        feature_subset: List[int],
        epochs: int = 50,
        lr: float = 0.001,
        verbose: bool = False
    ) -> float:
        """
        Train model on a specific subset of features and return performance.
        
        This function represents E_S in the Shapley value formula, computing
        the model's performance when trained with only the features in subset S.
        
        Args:
            feature_subset: List of feature indices to use
            epochs: Number of training epochs
            lr: Learning rate for optimization
            verbose: Whether to print training progress
            
        Returns:
            Performance metric value (lower is better for MSE/MAE)
        """
        # Handle empty subset case (all features missing)
        # This represents z_0 = E(h_s(0)) in the paper
        if len(feature_subset) == 0:
            # Return baseline performance (predict mean of training data)
            baseline_pred = self.Y_train.mean(dim=0, keepdim=True).expand_as(self.Y_train)
            return self.metric_fn(self.Y_train, baseline_pred)
        
        # Extract only the selected features from training data
        X_subset = self.X_train[:, feature_subset]
        
        # Create a new model instance for this feature subset
        # We need a fresh model to avoid bias from previous training
        subset_model = self._create_subset_model(len(feature_subset))
        
        # Define optimizer for training (Adam for fast convergence)
        optimizer = torch.optim.Adam(subset_model.parameters(), lr=lr)
        
        # Define loss function (MSE for regression)
        criterion = torch.nn.MSELoss()
        
        # Training loop for the subset model
        subset_model.train()
        for epoch in range(epochs):
            # Forward pass: compute predictions
            predictions = subset_model(X_subset)
            
            # Compute loss between predictions and ground truth
            loss = criterion(predictions, self.Y_train)
            
            # Backward pass: compute gradients
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute new gradients
            
            # Update model parameters
            optimizer.step()
            
            # Print progress if verbose
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Evaluate final model performance
        subset_model.eval()
        with torch.no_grad():
            final_predictions = subset_model(X_subset)
            performance = self.metric_fn(self.Y_train, final_predictions)
        
        return performance
    
    def _create_subset_model(self, n_input_features: int) -> torch.nn.Module:
        """
        Create a small neural network for subset evaluation.
        
        This model should be simple to train quickly for many feature subsets.
        
        Args:
            n_input_features: Number of input features for this subset
            
        Returns:
            PyTorch model instance
        """
        # Simple 2-layer fully connected network
        # Hidden layer size scales with input dimension
        hidden_size = max(32, n_input_features * 2)
        output_size = self.Y_train.shape[1]
        
        # Create sequential model: input -> hidden -> output
        model = torch.nn.Sequential(
            # First layer: project features to hidden space
            torch.nn.Linear(n_input_features, hidden_size),
            torch.nn.ReLU(),  # Non-linear activation
            
            # Output layer: project to target dimension
            torch.nn.Linear(hidden_size, output_size)
        ).to(self.device)
        
        return model
    
    def _powerset(self, features: List[int]) -> List[List[int]]:
        """
        Generate all subsets of a feature set (power set).
        
        For Shapley value computation, we need to evaluate the model on
        all possible feature subsets excluding the feature of interest.
        
        Args:
            features: List of feature indices
            
        Returns:
            List of all possible subsets (including empty set)
        """
        # Use itertools to generate all combinations of all sizes
        # chain combines results from different subset sizes
        return [list(subset) for subset in chain.from_iterable(
            combinations(features, r) for r in range(len(features) + 1)
        )]
    
    def compute_shapley_values(
        self,
        n_samples: int = None,
        epochs_per_subset: int = 50,
        use_approximation: bool = True,
        verbose: bool = True
    ) -> Dict[int, float]:
        """
        Compute Shapley values for all features.
        
        This implements Equation (11) from the paper:
        φ_a = Σ_{S⊆Z\\{a}} [|S|!(Z-|S|-1)!/Z!] * [E_{S∪{a}} - E_S]
        
        Args:
            n_samples: Number of subset samples (for approximation)
            epochs_per_subset: Training epochs for each subset model
            use_approximation: Use sampling approximation for large feature sets
            verbose: Show progress bar
            
        Returns:
            Dictionary mapping feature index to Shapley value
        """
        # Initialize storage for Shapley values (one per feature)
        shapley_values = {i: 0.0 for i in range(self.n_features)}
        
        # For small feature sets, compute exact Shapley values
        # For large sets, use Monte Carlo approximation to reduce computation
        if not use_approximation or self.n_features <= 10:
            # Exact computation: iterate over all features
            if verbose:
                print(f"Computing exact Shapley values for {self.n_features} features...")
            
            # Create iterator with progress bar if verbose
            feature_iterator = tqdm(range(self.n_features)) if verbose else range(self.n_features)
            
            for feature_idx in feature_iterator:
                # Get all other features (excluding current feature)
                other_features = [i for i in range(self.n_features) if i != feature_idx]
                
                # Generate all subsets of other features
                # These represent all possible contexts S in the Shapley formula
                all_subsets = self._powerset(other_features)
                
                # Compute marginal contribution across all subsets
                for subset in all_subsets:
                    # Compute model performance without current feature: E_S
                    perf_without = self._train_subset_model(
                        subset,
                        epochs=epochs_per_subset,
                        verbose=False
                    )
                    
                    # Compute model performance with current feature: E_{S∪{a}}
                    perf_with = self._train_subset_model(
                        subset + [feature_idx],
                        epochs=epochs_per_subset,
                        verbose=False
                    )
                    
                    # Compute marginal contribution: E_{S∪{a}} - E_S
                    # For error metrics (MSE/MAE), lower is better, so we negate
                    marginal_contribution = perf_without - perf_with
                    
                    # Compute Shapley weight: |S|!(M-|S|-1)!/M!
                    # This weight ensures fair attribution across subset sizes
                    subset_size = len(subset)
                    M = self.n_features
                    
                    # Compute factorials for Shapley weight
                    weight = (
                        np.math.factorial(subset_size) *
                        np.math.factorial(M - subset_size - 1) /
                        np.math.factorial(M)
                    )
                    
                    # Add weighted contribution to Shapley value
                    shapley_values[feature_idx] += weight * marginal_contribution
                    
        else:
            # Approximation for large feature sets using Monte Carlo sampling
            # Sample random subsets instead of evaluating all 2^M subsets
            if verbose:
                print(f"Using Monte Carlo approximation with {n_samples} samples...")
            
            # Default number of samples if not specified
            if n_samples is None:
                n_samples = min(1000, 2 ** self.n_features)
            
            # Iterate over all features
            for feature_idx in tqdm(range(self.n_features)) if verbose else range(self.n_features):
                # Sample random subsets for approximation
                for _ in range(n_samples):
                    # Get other features (excluding current one)
                    other_features = [i for i in range(self.n_features) if i != feature_idx]
                    
                    # Randomly sample a subset of other features
                    # Subset size drawn uniformly from 0 to M-1
                    subset_size = np.random.randint(0, len(other_features) + 1)
                    subset = list(np.random.choice(
                        other_features,
                        size=subset_size,
                        replace=False
                    ))
                    
                    # Compute marginal contribution for this sampled subset
                    perf_without = self._train_subset_model(
                        subset,
                        epochs=epochs_per_subset,
                        verbose=False
                    )
                    
                    perf_with = self._train_subset_model(
                        subset + [feature_idx],
                        epochs=epochs_per_subset,
                        verbose=False
                    )
                    
                    # Marginal contribution (improvement when adding this feature)
                    marginal_contribution = perf_without - perf_with
                    
                    # For Monte Carlo, we average contributions without explicit weights
                    # The sampling distribution implicitly provides proper weighting
                    shapley_values[feature_idx] += marginal_contribution / n_samples
        
        # Store computed values
        self.shapley_values = shapley_values
        
        return shapley_values
    
    def get_top_features(
        self,
        k: int = None,
        threshold: float = None
    ) -> List[int]:
        """
        Select the most important features based on Shapley values.
        
        This implements the feature selection mechanism described in the paper,
        where only the most informative features are used for prediction.
        
        Args:
            k: Number of top features to select
            threshold: Minimum Shapley value threshold for selection
            
        Returns:
            List of selected feature indices (sorted by importance)
        """
        if self.shapley_values is None:
            raise ValueError("Must call compute_shapley_values() first")
        
        # Sort features by Shapley value (descending order)
        sorted_features = sorted(
            self.shapley_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if k is not None:
            # Select top k features by importance
            return [feat_idx for feat_idx, _ in sorted_features[:k]]
            
        elif threshold is not None:
            # Select features above importance threshold
            return [feat_idx for feat_idx, value in sorted_features if value > threshold]
            
        else:
            # Return all features sorted by importance
            return [feat_idx for feat_idx, _ in sorted_features]
    
    def get_feature_importance_dict(self) -> Dict[int, float]:
        """
        Get a dictionary of feature importances.
        
        Returns:
            Dictionary mapping feature index to Shapley value
        """
        if self.shapley_values is None:
            raise ValueError("Must call compute_shapley_values() first")
            
        return self.shapley_values.copy()
    
    def visualize_importance(self, feature_names: List[str] = None) -> None:
        """
        Print a visual representation of feature importances.
        
        Args:
            feature_names: Optional list of feature names for display
        """
        if self.shapley_values is None:
            raise ValueError("Must call compute_shapley_values() first")
        
        # Sort features by importance
        sorted_items = sorted(
            self.shapley_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print header
        print("\nFeature Importance (Shapley Values):")
        print("=" * 60)
        
        # Print each feature with importance bar
        max_value = max(abs(v) for v in self.shapley_values.values())
        for feat_idx, value in sorted_items:
            # Get feature name or use index
            name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
            
            # Create visual bar (normalized to max value)
            bar_length = int(40 * abs(value) / max_value) if max_value > 0 else 0
            bar = '█' * bar_length
            
            # Print feature importance with bar
            print(f"{name:20s} | {value:8.4f} | {bar}")
        
        print("=" * 60)


def select_features_with_shapley(
    model: torch.nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    k: int = None,
    threshold: float = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[int], Dict[int, float]]:
    """
    Convenience function to compute Shapley values and select features.
    
    This encapsulates the feature selection process described in the paper:
    1. Compute Shapley values for all features
    2. Select the most informative features based on their contribution
    
    Args:
        model: PyTorch model for evaluation
        X_train: Training features, shape (n_samples, n_features)
        Y_train: Training targets, shape (n_samples, output_dim)
        k: Number of top features to select (optional)
        threshold: Minimum Shapley value threshold (optional)
        device: Computation device
        
    Returns:
        selected_features: List of selected feature indices
        importance_dict: Dictionary of all Shapley values
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X_train).float()
    Y_tensor = torch.from_numpy(Y_train).float()
    
    # Initialize Shapley calculator
    sfv = ShapleyFeatureValue(
        model=model,
        X_train=X_tensor,
        Y_train=Y_tensor,
        metric='mse',
        device=device
    )
    
    # Compute Shapley values for all features
    print("Computing Shapley Feature Values...")
    importance_dict = sfv.compute_shapley_values(
        use_approximation=True,
        n_samples=100,  # Use sampling for efficiency
        verbose=True
    )
    
    # Select top features
    selected_features = sfv.get_top_features(k=k, threshold=threshold)
    
    # Display results
    sfv.visualize_importance()
    
    return selected_features, importance_dict


if __name__ == "__main__":
    """
    Example usage of Shapley Feature Value for feature selection.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic haptic data
    n_samples = 200
    n_features = 9  # 3 DoF × 3 measurements
    output_dim = 9
    
    # Create synthetic data where only some features are informative
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Make first 3 features highly predictive
    Y_train = (X_train[:, :3] @ np.random.randn(3, output_dim).astype(np.float32) +
               0.1 * np.random.randn(n_samples, output_dim).astype(np.float32))
    
    # Create a simple model for evaluation
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, output_dim)
    )
    
    # Select features using Shapley values
    selected_features, importance = select_features_with_shapley(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        k=5  # Select top 5 features
    )
    
    print(f"\nSelected features: {selected_features}")
    print(f"\nTop 3 most important features should be 0, 1, 2")
    print("(since we constructed Y from the first 3 features)")
