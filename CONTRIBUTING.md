# Contributing to GP+SFV Framework

Thank you for your interest in contributing to our project! This document provides guidelines for contributing to the Shapley Features for Robust Signal Prediction in Tactile Internet framework.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git for version control
- Familiarity with Gaussian Processes and deep learning

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction.git
   cd Gaussian-Process-Shapley-Feature-Value-for-Signal-Prediction
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies (including development tools):
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Making Changes

1. **Always work on a feature branch**: Never commit directly to `main`
2. **Keep commits atomic**: Each commit should represent a single logical change
3. **Write descriptive commit messages**: Follow the format:
   ```
   [Component] Brief description
   
   Detailed explanation of what changed and why
   ```

### Example Workflow

```bash
# Create feature branch
git checkout -b feature/add-new-kernel

# Make changes to files
# ... edit code ...

# Add and commit changes
git add gaussian_process.py
git commit -m "[GP] Add Matérn 5/2 kernel option

Extends kernel options to include Matérn 5/2 for smoother
interpolation. Maintains backward compatibility with existing code."

# Push to your fork
git push origin feature/add-new-kernel
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped in order (standard library, third-party, local)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`
  - Private methods: `_leading_underscore`

### Comments and Documentation

Every function, class, and module should have detailed comments:

```python
def compute_shapley_value(
    features: List[int],
    model: nn.Module,
    data: np.ndarray
) -> float:
    """
    Compute Shapley value for a feature subset.
    
    This function implements Equation (11) from the paper,
    computing the marginal contribution of features.
    
    Args:
        features: List of feature indices to evaluate
        model: PyTorch model for evaluation
        data: Input data array, shape (n_samples, n_features)
        
    Returns:
        Shapley value (float)
        
    Raises:
        ValueError: If features list is empty
        
    Example:
        >>> features = [0, 1, 2]
        >>> shapley_val = compute_shapley_value(features, model, X_train)
        >>> print(f"Shapley value: {shapley_val:.4f}")
    """
    # Implementation with detailed inline comments
    # ...
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Tuple, Optional

def process_data(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process input data."""
    # ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_gaussian_process.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_gaussian_process.py
import pytest
import torch
import numpy as np
from gaussian_process import GaussianProcess

def test_gp_prediction():
    """Test GP prediction accuracy."""
    # Setup test data
    X_train = np.random.randn(100, 9).astype(np.float32)
    Y_train = np.random.randn(100, 9).astype(np.float32)
    
    # Create and fit GP
    gp = GaussianProcess()
    gp.fit(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    
    # Test prediction
    X_test = np.random.randn(10, 9).astype(np.float32)
    mean, std = gp.predict(torch.from_numpy(X_test), return_std=True)
    
    # Assertions
    assert mean.shape == (10, 9)
    assert std.shape == (10, 9)
    assert torch.all(std > 0)  # Uncertainty should be positive
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    Brief description of what the function does.
    
    More detailed explanation if needed, including:
    - Important algorithmic details
    - References to paper equations
    - Edge cases and limitations
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this exception occurs
        TypeError: When this exception occurs
        
    Example:
        >>> result = function_name(42, "test")
        >>> print(result)
        True
    """
```

### README Updates

When adding new features, update the README.md:
- Add to feature list
- Update examples if applicable
- Add to table of contents

## Submitting Changes

### Pull Request Process

1. **Update documentation**: Ensure all docstrings are complete
2. **Add tests**: New features should include unit tests
3. **Update CHANGELOG**: Add entry describing your changes
4. **Run tests**: Ensure all tests pass locally
5. **Check code style**: Run linting tools

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Related Issue
Fixes #(issue number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] No breaking changes (or documented if unavoidable)
```

### Review Process

1. A maintainer will review your PR within 1-2 weeks
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Celebrate! 🎉 Your contribution is now part of the project

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Additional kernel functions for GP
- More efficient Shapley value approximations
- Support for additional datasets
- Performance optimizations
- Bug fixes

### Medium Priority
- Improved visualization tools
- Additional neural network architectures
- Hyperparameter tuning utilities
- Better documentation and examples

### Future Enhancements
- Real-time inference optimizations
- Integration with ROS for robotics
- Web-based demo interface
- Mobile deployment support

## Questions?

If you have questions about contributing:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Contact the maintainers: av@ece.au.dk, qz@ece.au.dk

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to advancing Tactile Internet research!
