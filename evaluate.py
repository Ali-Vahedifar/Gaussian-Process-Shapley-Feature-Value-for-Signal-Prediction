"""
Model Evaluation Script
=======================
This script evaluates trained GP+SFV models on test data and generates
comprehensive performance metrics and visualizations.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import json
import os
from tqdm import tqdm

# Import custom modules
from gaussian_process import GaussianProcess
from models import create_model
from data_utils import load_haptic_dataset, create_dataloaders


class ModelEvaluator:
    """
    Comprehensive model evaluation and metrics computation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        gp_oracle: GaussianProcess,
        selected_features: List[int],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained neural network
            gp_oracle: GP oracle for reference predictions
            selected_features: Feature indices selected by SFV
            device: Computation device
        """
        # Store components
        self.model = model.to(device)
        self.gp_oracle = gp_oracle
        self.selected_features = selected_features
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def compute_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        tolerance: float = 0.1
    ) -> float:
        """
        Compute prediction accuracy (percentage within tolerance).
        
        Args:
            predictions: Model predictions, shape (n_samples, output_dim)
            targets: Ground truth targets, shape (n_samples, output_dim)
            tolerance: Acceptable error threshold (default 10%)
            
        Returns:
            Accuracy as percentage (0-100)
        """
        # Compute relative error
        relative_error = np.abs((predictions - targets) / (np.abs(targets) + 1e-8))
        
        # Count predictions within tolerance
        within_tolerance = (relative_error < tolerance).astype(float)
        
        # Average across all samples and features
        accuracy = np.mean(within_tolerance) * 100
        
        return accuracy
    
    def compute_metrics(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing all metrics
        """
        # Initialize storage for predictions and targets
        all_predictions = []
        all_targets = []
        all_gp_predictions = []
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            # Iterate through test batches
            for X_batch, Y_batch in tqdm(test_loader, desc="Computing metrics"):
                # Move to device
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Select features
                X_selected = X_batch[:, self.selected_features]
                
                # Get model predictions
                predictions = self.model(X_selected)
                
                # Get GP predictions (for comparison)
                gp_mean, _ = self.gp_oracle.predict(X_batch, return_std=False)
                
                # Store results (move to CPU and convert to numpy)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(Y_batch.cpu().numpy())
                all_gp_predictions.append(gp_mean.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        gp_predictions = np.vstack(all_gp_predictions)
        
        # Compute various metrics
        metrics = {}
        
        # 1. Mean Squared Error (MSE)
        mse = np.mean((predictions - targets) ** 2)
        metrics['mse'] = float(mse)
        
        # 2. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        metrics['rmse'] = float(rmse)
        
        # 3. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - targets))
        metrics['mae'] = float(mae)
        
        # 4. Accuracy (within 10% tolerance)
        accuracy = self.compute_accuracy(predictions, targets, tolerance=0.1)
        metrics['accuracy'] = float(accuracy)
        
        # 5. R-squared (coefficient of determination)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2_score'] = float(r2)
        
        # 6. Per-feature metrics
        # Compute accuracy for each of the 9 features separately
        feature_names = [
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'force_x', 'force_y', 'force_z'
        ]
        
        for i, name in enumerate(feature_names):
            if i < predictions.shape[1]:
                # Accuracy for this feature
                feat_acc = self.compute_accuracy(
                    predictions[:, i:i+1],
                    targets[:, i:i+1],
                    tolerance=0.1
                )
                metrics[f'accuracy_{name}'] = float(feat_acc)
                
                # MSE for this feature
                feat_mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
                metrics[f'mse_{name}'] = float(feat_mse)
        
        # 7. Comparison with GP oracle
        # How close is NN to GP predictions?
        gp_alignment_mse = np.mean((predictions - gp_predictions) ** 2)
        metrics['gp_alignment_mse'] = float(gp_alignment_mse)
        
        return metrics
    
    def plot_predictions(
        self,
        test_loader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        save_path: str = 'predictions.png'
    ) -> None:
        """
        Plot predicted vs actual values for visualization.
        
        Args:
            test_loader: DataLoader for test data
            n_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        # Get first batch of predictions
        with torch.no_grad():
            X_batch, Y_batch = next(iter(test_loader))
            X_batch = X_batch[:n_samples].to(self.device)
            Y_batch = Y_batch[:n_samples].to(self.device)
            
            # Get predictions
            X_selected = X_batch[:, self.selected_features]
            predictions = self.model(X_selected)
            
            # Move to CPU for plotting
            predictions = predictions.cpu().numpy()
            targets = Y_batch.cpu().numpy()
        
        # Create figure with subplots for each feature
        feature_names = [
            'Pos X', 'Pos Y', 'Pos Z',
            'Vel X', 'Vel Y', 'Vel Z',
            'Force X', 'Force Y', 'Force Z'
        ]
        
        n_features = min(9, predictions.shape[1])
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            # Scatter plot: predicted vs actual
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=20)
            
            # Perfect prediction line (diagonal)
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Actual', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Overall title
        plt.suptitle('Predicted vs Actual Values', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction plot saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(
        self,
        test_loader: torch.utils.data.DataLoader,
        save_path: str = 'error_distribution.png'
    ) -> None:
        """
        Plot distribution of prediction errors.
        
        Args:
            test_loader: DataLoader for test data
            save_path: Path to save the plot
        """
        # Collect all predictions and errors
        all_errors = []
        
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Get predictions
                X_selected = X_batch[:, self.selected_features]
                predictions = self.model(X_selected)
                
                # Compute errors
                errors = (predictions - Y_batch).cpu().numpy()
                all_errors.append(errors)
        
        # Concatenate all errors
        all_errors = np.vstack(all_errors)
        
        # Create figure with error distributions for each feature
        feature_names = [
            'Pos X', 'Pos Y', 'Pos Z',
            'Vel X', 'Vel Y', 'Vel Z',
            'Force X', 'Force Y', 'Force Z'
        ]
        
        n_features = min(9, all_errors.shape[1])
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            # Histogram of errors
            ax.hist(all_errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            
            # Add vertical line at zero (perfect prediction)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Prediction Error', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature_names[i]} Error Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_error = np.mean(all_errors[:, i])
            std_error = np.std(all_errors[:, i])
            ax.text(0.02, 0.98, f'μ = {mean_error:.4f}\nσ = {std_error:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        plt.suptitle('Prediction Error Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
        plt.close()


def main():
    """
    Main evaluation function.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate GP+SFV model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='drag_max_stiffness_y',
                        help='Name of dataset to evaluate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, or cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint['config']
    selected_features = checkpoint['selected_features']
    
    print(f"Model configuration:")
    print(f"  Architecture: {config['architecture']}")
    print(f"  Selected features: {selected_features}")
    
    # Load test data
    print(f"\nLoading test dataset: {args.dataset}")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_haptic_dataset(
        dataset_name=args.dataset
    )
    
    # Create test loader
    _, _, test_loader = create_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=args.batch_size
    )
    
    # Recreate GP oracle (needed for evaluation)
    print("\nRecreating GP oracle...")
    from gaussian_process import create_gp_oracle
    gp_oracle = create_gp_oracle(X_train, Y_train, device=device)
    
    # Recreate model
    print("Recreating neural network model...")
    model = create_model(
        architecture=config['architecture'],
        input_dim=len(selected_features),
        output_dim=config['output_dim'],
        device=device
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        gp_oracle=gp_oracle,
        selected_features=selected_features,
        device=device
    )
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = evaluator.compute_metrics(test_loader)
    
    # Print main metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  GP Alignment MSE: {metrics['gp_alignment_mse']:.6f}")
    
    # Print per-feature accuracy
    print(f"\nPer-Feature Accuracy:")
    feature_names = [
        'position_x', 'position_y', 'position_z',
        'velocity_x', 'velocity_y', 'velocity_z',
        'force_x', 'force_y', 'force_z'
    ]
    for name in feature_names:
        key = f'accuracy_{name}'
        if key in metrics:
            print(f"  {name:12s}: {metrics[key]:6.2f}%")
    
    # Save metrics to file
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics saved to evaluation_results/metrics.json")
    
    # Generate visualization plots
    print("\nGenerating visualization plots...")
    evaluator.plot_predictions(
        test_loader,
        n_samples=100,
        save_path='evaluation_results/predictions.png'
    )
    
    evaluator.plot_error_distribution(
        test_loader,
        save_path='evaluation_results/error_distribution.png'
    )
    
    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
