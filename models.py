"""
Neural Network Models for Tactile Internet Signal Prediction
=============================================================
This module implements the neural network architectures used in the paper:
- Fully Connected Network (FC)
- Long Short-Term Memory Network (LSTM)
- Residual Network (ResNet)

These networks are trained using Jensen-Shannon Divergence loss to learn
from the GP oracle's probabilistic predictions.

Reference: "Shapley Features for Robust Signal Prediction in Tactile Internet"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FullyConnectedNetwork(nn.Module):
    """
    Fully Connected Neural Network for haptic signal prediction.
    
    Architecture: 12 layers with 100 ReLU-activated units each.
    This is one of the three architectures evaluated in the paper.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 100,
        n_layers: int = 12,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Fully Connected Network.
        
        Args:
            input_dim: Number of input features (selected by SFV)
            output_dim: Number of output predictions (force, velocity, position)
            hidden_dim: Number of units in each hidden layer
            n_layers: Total number of layers
            dropout_rate: Dropout probability for regularization
        """
        super(FullyConnectedNetwork, self).__init__()
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create list to store all layers
        layers = []
        
        # First layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())  # Non-linear activation
        layers.append(nn.Dropout(dropout_rate))  # Regularization
        
        # Middle layers: hidden -> hidden (repeated n_layers - 2 times)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Final layer: hidden -> output (no activation for regression)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization (good for deep networks)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        This helps with gradient flow in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                # Initialize bias to small positive value
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.network(x)


class LSTMNetwork(nn.Module):
    """
    LSTM Neural Network for sequential haptic signal prediction.
    
    Architecture: 2 stacked LSTM layers with 128 units each, followed by
    a dense output layer. LSTMs are well-suited for time-series prediction
    as they can capture temporal dependencies in haptic signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize LSTM Network.
        
        Args:
            input_dim: Number of input features per timestep
            output_dim: Number of output predictions
            hidden_dim: Number of LSTM hidden units
            n_layers: Number of stacked LSTM layers
            dropout_rate: Dropout between LSTM layers
        """
        super(LSTMNetwork, self).__init__()
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create LSTM layers
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout_rate if n_layers > 1 else 0,  # Dropout between LSTM layers
            batch_first=True
        )
        
        # Dense output layer to project LSTM output to target dimension
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize LSTM and linear layer weights.
        """
        # Initialize LSTM weights using Xavier initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional tuple of (h_0, c_0) hidden states
            
        Returns:
            output: Predictions of shape (batch_size, seq_len, output_dim)
            hidden: Tuple of final (h_n, c_n) hidden states
        """
        # Pass through LSTM layers
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (n_layers, batch_size, hidden_dim) for both h and c
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Project LSTM output to target dimension
        # Apply linear layer to each timestep
        output = self.fc_out(lstm_out)
        
        return output, hidden
    
    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict only the next timestep (for online inference).
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Next prediction of shape (batch_size, output_dim)
        """
        # Get full sequence predictions
        output, _ = self.forward(x)
        
        # Return only the last timestep prediction
        return output[:, -1, :]


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet architecture.
    
    Implements skip connections: output = F(x) + x
    This helps with gradient flow in deep networks and enables
    learning of residual (difference) mappings.
    """
    
    def __init__(
        self,
        dim: int,
        dropout_rate: float = 0.1
    ):
        """
        Initialize residual block.
        
        Args:
            dim: Dimension of input/output (must be same for skip connection)
            dropout_rate: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        
        # First linear transformation
        self.fc1 = nn.Linear(dim, dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second linear transformation
        self.fc2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Final ReLU applied after skip connection
        self.relu2 = nn.ReLU()
        
        # Initialize weights using He initialization (good for ReLU)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Implements: output = ReLU(F(x) + x) where F(x) = fc2(relu(fc1(x)))
        
        Args:
            x: Input tensor of shape (batch_size, dim)
            
        Returns:
            Output tensor of shape (batch_size, dim)
        """
        # Store input for skip connection
        identity = x
        
        # First transformation: x -> fc1 -> relu -> dropout
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second transformation: -> fc2 -> dropout
        out = self.fc2(out)
        out = self.dropout2(out)
        
        # Add skip connection (residual)
        out = out + identity
        
        # Final activation
        out = self.relu2(out)
        
        return out


class ResNet(nn.Module):
    """
    Residual Network (ResNet-32) for haptic signal prediction.
    
    ResNets use skip connections to enable training of very deep networks
    without gradient vanishing. This architecture achieved the best results
    in the paper when combined with GP+SFV.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 8,
        dropout_rate: float = 0.1
    ):
        """
        Initialize ResNet.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output predictions
            hidden_dim: Dimension of residual blocks
            n_blocks: Number of residual blocks
            dropout_rate: Dropout probability
        """
        super(ResNet, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        
        # Input projection: project input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(n_blocks)
        ])
        
        # Output projection: project hidden to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        This is recommended for networks with ReLU activations.
        """
        # Initialize input projection
        nn.init.kaiming_uniform_(self.input_proj[0].weight, nonlinearity='relu')
        nn.init.constant_(self.input_proj[0].bias, 0.01)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        # Project input to hidden dimension
        out = self.input_proj(x)
        
        # Pass through each residual block
        for block in self.residual_blocks:
            out = block(out)
        
        # Project to output dimension
        out = self.output_proj(out)
        
        return out


def create_model(
    architecture: str,
    input_dim: int,
    output_dim: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """
    Factory function to create neural network models.
    
    This function provides a unified interface for creating the three
    architectures evaluated in the paper: FC, LSTM, and ResNet.
    
    Args:
        architecture: Model type ('fc', 'lstm', or 'resnet')
        input_dim: Number of input features (after SFV selection)
        output_dim: Number of output predictions
        device: Computation device
        
    Returns:
        Initialized PyTorch model
    """
    # Convert architecture name to lowercase for case-insensitive matching
    arch = architecture.lower()
    
    if arch == 'fc':
        # Create Fully Connected Network
        model = FullyConnectedNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=100,  # As specified in paper
            n_layers=12,      # As specified in paper
            dropout_rate=0.1
        )
        
    elif arch == 'lstm':
        # Create LSTM Network
        model = LSTMNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=128,   # As specified in paper
            n_layers=2,       # As specified in paper
            dropout_rate=0.1
        )
        
    elif arch == 'resnet':
        # Create ResNet (best performing in paper)
        model = ResNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=256,   # Can be tuned
            n_blocks=8,       # ResNet-32 equivalent
            dropout_rate=0.1
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'fc', 'lstm', or 'resnet'")
    
    # Move model to specified device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nCreated {architecture.upper()} model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


if __name__ == "__main__":
    """
    Test the neural network architectures.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    input_dim = 9   # 9 features (after SFV selection)
    output_dim = 9  # Predict 9 outputs (3 DoF × 3 measurements)
    batch_size = 32
    
    # Create sample input
    x = torch.randn(batch_size, input_dim)
    
    print("Testing Neural Network Architectures")
    print("=" * 60)
    
    # Test Fully Connected Network
    print("\n1. Fully Connected Network:")
    fc_model = create_model('fc', input_dim, output_dim)
    fc_out = fc_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {fc_out.shape}")
    
    # Test LSTM Network (needs sequence input)
    print("\n2. LSTM Network:")
    lstm_model = create_model('lstm', input_dim, output_dim)
    x_seq = x.unsqueeze(1)  # Add sequence dimension: (batch, 1, features)
    lstm_out, _ = lstm_model(x_seq)
    print(f"   Input shape: {x_seq.shape}")
    print(f"   Output shape: {lstm_out.shape}")
    
    # Test ResNet
    print("\n3. ResNet:")
    resnet_model = create_model('resnet', input_dim, output_dim)
    resnet_out = resnet_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {resnet_out.shape}")
    
    print("\n" + "=" * 60)
    print("All models working correctly!")
