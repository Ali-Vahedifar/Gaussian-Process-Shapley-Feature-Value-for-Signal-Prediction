"""
Training Script for GP+SFV Framework
=====================================
This script implements the complete training pipeline from the paper:
1. Train GP oracle on historical haptic data
2. Compute Shapley Feature Values for feature selection
3. Train neural network using JSD loss with GP guidance

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import custom modules
from gaussian_process import GaussianProcess, create_gp_oracle
from shapley_feature_value import ShapleyFeatureValue
from models import create_model
from loss_functions import create_loss_function
from data_utils import load_haptic_dataset, create_dataloaders, TactileDataset


class GPSFVTrainer:
    """
    Main trainer class for GP+SFV framework.
    
    Implements the two-stage training process:
    1. Offline training: GP oracle + SFV feature selection
    2. Online training: Neural network training with JSD loss
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary containing all hyperparameters
            device: Computation device
        """
        # Store configuration
        self.config = config
        self.device = device
        
        # Initialize placeholders for models and data
        self.gp_oracle = None          # Gaussian Process oracle
        self.selected_features = None   # Features selected by SFV
        self.model = None              # Neural network model
        self.optimizer = None          # Optimizer for NN training
        self.loss_fn = None           # Loss function (JSD)
        
        # Training history storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_jsd': [],
            'train_mse': [],
        }
        
    def stage1_train_gp_oracle(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray
    ) -> GaussianProcess:
        """
        Stage 1a: Train Gaussian Process oracle on historical data.
        
        The GP learns the probability distribution of haptic signals and
        serves as ground truth for neural network training.
        
        Args:
            X_train: Historical input signals, shape (n_samples, n_features)
            Y_train: Corresponding target signals, shape (n_samples, output_dim)
            
        Returns:
            Trained GP oracle
        """
        print("\n" + "=" * 60)
        print("STAGE 1a: Training Gaussian Process Oracle")
        print("=" * 60)
        
        # Create and fit GP oracle
        self.gp_oracle = create_gp_oracle(
            X_history=X_train,
            Y_history=Y_train,
            kernel_type=self.config['gp_kernel'],
            device=self.device
        )
        
        # Compute log marginal likelihood (quality metric)
        log_ml = self.gp_oracle.log_marginal_likelihood()
        print(f"GP Log Marginal Likelihood: {log_ml:.4f}")
        
        # Generate predictions with uncertainty for validation
        X_test_sample = torch.from_numpy(X_train[:10]).float().to(self.device)
        mean, std = self.gp_oracle.predict(X_test_sample, return_std=True)
        
        print(f"GP Prediction Statistics:")
        print(f"  Mean prediction range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"  Mean uncertainty: {std.mean():.4f} ± {std.std():.4f}")
        print(f"  Uncertainty range: [{std.min():.4f}, {std.max():.4f}]")
        
        return self.gp_oracle
    
    def stage1_compute_shapley_values(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        k_features: int = None
    ) -> list:
        """
        Stage 1b: Compute Shapley Feature Values for feature selection.
        
        SFV identifies the most informative features for prediction,
        improving both accuracy and computational efficiency.
        
        Args:
            X_train: Training features
            Y_train: Training targets
            k_features: Number of top features to select
            
        Returns:
            List of selected feature indices
        """
        print("\n" + "=" * 60)
        print("STAGE 1b: Computing Shapley Feature Values")
        print("=" * 60)
        
        # Convert data to tensors
        X_tensor = torch.from_numpy(X_train).float()
        Y_tensor = torch.from_numpy(Y_train).float()
        
        # Create a simple model for SFV evaluation
        # This model is only for feature selection, not final predictions
        eval_model = create_model(
            architecture='fc',  # Use simple FC for efficiency
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            device=self.device
        )
        
        # Initialize SFV calculator
        sfv = ShapleyFeatureValue(
            model=eval_model,
            X_train=X_tensor,
            Y_train=Y_tensor,
            metric='mse',
            device=self.device
        )
        
        # Compute Shapley values for all features
        # Use approximation for large feature sets (>10 features)
        shapley_values = sfv.compute_shapley_values(
            use_approximation=(X_train.shape[1] > 10),
            n_samples=self.config.get('sfv_samples', 100),
            epochs_per_subset=self.config.get('sfv_epochs', 30),
            verbose=True
        )
        
        # Visualize feature importance
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
        sfv.visualize_importance(feature_names)
        
        # Select top k features
        if k_features is None:
            k_features = self.config.get('n_selected_features', X_train.shape[1] // 2)
        
        self.selected_features = sfv.get_top_features(k=k_features)
        
        print(f"\nSelected {len(self.selected_features)} features: {self.selected_features}")
        
        return self.selected_features
    
    def stage2_train_neural_network(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int
    ) -> nn.Module:
        """
        Stage 2: Train neural network with GP oracle guidance.
        
        The NN learns to mimic the GP's probability distribution using
        Jensen-Shannon Divergence loss, enabling fast real-time inference.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            n_epochs: Number of training epochs
            
        Returns:
            Trained neural network model
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Training Neural Network with GP Guidance")
        print("=" * 60)
        
        # Create model with selected features
        n_features = len(self.selected_features)
        output_dim = self.config['output_dim']
        
        self.model = create_model(
            architecture=self.config['architecture'],
            input_dim=n_features,
            output_dim=output_dim,
            device=self.device
        )
        
        # Initialize optimizer (SGD with momentum as in paper)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config.get('weight_decay', 0.0)
        )
        
        # Learning rate scheduler (reduces LR on plateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Initialize loss function (JSD as in paper)
        self.loss_fn = create_loss_function(
            loss_type=self.config['loss_type'],
            reduction='mean'
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate based on validation loss
            scheduler.step(val_metrics['loss'])
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            if 'jsd' in train_metrics:
                self.history['train_jsd'].append(train_metrics['jsd'])
                self.history['train_mse'].append(train_metrics['mse'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self._save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.get('patience', 20):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self._load_checkpoint('best_model.pth')
        
        return self.model
    
    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        total_loss = 0.0
        total_jsd = 0.0
        total_mse = 0.0
        n_batches = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (X_batch, Y_batch) in enumerate(pbar):
            # Move data to device
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            # Select features using SFV selection
            X_selected = X_batch[:, self.selected_features]
            
            # Get GP oracle predictions (ground truth distribution)
            with torch.no_grad():
                gp_mean, gp_std = self.gp_oracle.predict(X_batch, return_std=True)
            
            # Forward pass through neural network
            nn_predictions = self.model(X_selected)
            
            # For JSD loss, we need to estimate NN prediction uncertainty
            # Use dropout-based uncertainty estimation
            # (In production, you might want a more sophisticated method)
            nn_mean = nn_predictions
            
            # Estimate uncertainty using prediction variance
            # (Simple heuristic: use a learned or fixed uncertainty)
            nn_std = torch.ones_like(nn_mean) * 0.1  # Fixed for simplicity
            
            # Compute loss
            if isinstance(self.loss_fn, nn.MSELoss):
                # Simple MSE loss (baseline)
                loss = self.loss_fn(nn_mean, gp_mean)
                jsd_component = torch.tensor(0.0)
                mse_component = loss
            else:
                # JSD or combined loss
                loss, jsd_component, mse_component = self.loss_fn(
                    nn_mean, nn_std, gp_mean, gp_std
                )
            
            # Backward pass
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()             # Compute new gradients
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_jsd += jsd_component.item() if not isinstance(jsd_component, float) else 0
            total_mse += mse_component.item() if not isinstance(mse_component, float) else 0
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / n_batches
            })
        
        # Return average metrics
        return {
            'loss': total_loss / n_batches,
            'jsd': total_jsd / n_batches,
            'mse': total_mse / n_batches
        }
    
    def _validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        n_batches = 0
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Select features
                X_selected = X_batch[:, self.selected_features]
                
                # Get GP predictions
                gp_mean, gp_std = self.gp_oracle.predict(X_batch, return_std=True)
                
                # Forward pass
                nn_predictions = self.model(X_selected)
                nn_mean = nn_predictions
                nn_std = torch.ones_like(nn_mean) * 0.1
                
                # Compute loss
                if isinstance(self.loss_fn, nn.MSELoss):
                    loss = self.loss_fn(nn_mean, gp_mean)
                else:
                    loss, _, _ = self.loss_fn(nn_mean, nn_std, gp_mean, gp_std)
                
                # Accumulate metrics
                total_loss += loss.item()
                n_batches += 1
        
        return {'loss': total_loss / n_batches}
    
    def _save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'selected_features': self.selected_features,
            'history': self.history
        }
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join('checkpoints', filename))
    
    def _load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = torch.load(os.path.join('checkpoints', filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    """
    Main training function with command-line arguments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train GP+SFV model for TI')
    parser.add_argument('--dataset', type=str, default='drag_max_stiffness_y',
                        help='Name of dataset to use')
    parser.add_argument('--architecture', type=str, default='resnet',
                        choices=['fc', 'lstm', 'resnet'],
                        help='Neural network architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--n_features', type=int, default=5,
                        help='Number of features to select with SFV')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\nUsing device: {device}")
    
    # Configuration dictionary
    config = {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'gp_kernel': 'rbf',
        'learning_rate': args.lr,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'batch_size': args.batch_size,
        'n_selected_features': args.n_features,
        'loss_type': 'combined',  # Use combined JSD+MSE loss
        'patience': 20,
        'sfv_samples': 100,
        'sfv_epochs': 30,
        'output_dim': 9  # 3 DoF × 3 measurements
    }
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_haptic_dataset(
        dataset_name=args.dataset
    )
    
    print(f"Data shapes:")
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=config['batch_size']
    )
    
    # Initialize trainer
    trainer = GPSFVTrainer(config=config, device=device)
    
    # Stage 1: Train GP oracle and compute SFV
    trainer.stage1_train_gp_oracle(X_train, Y_train)
    trainer.stage1_compute_shapley_values(X_train, Y_train, k_features=args.n_features)
    
    # Stage 2: Train neural network
    model = trainer.stage2_train_neural_network(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs
    )
    
    # Save final results
    results = {
        'config': config,
        'selected_features': trainer.selected_features,
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_loss': trainer.history['val_loss'][-1],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Final validation loss: {results['final_val_loss']:.6f}")
    print(f"Selected features: {results['selected_features']}")
    print("\nResults saved to results.json")
    print("Model checkpoint saved to checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
