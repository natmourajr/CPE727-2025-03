# Models

PyTorch model architectures for the CPE727 Deep Learning course.

## Available Models

### SimpleMLP
A basic multi-layer perceptron with one hidden layer.

**Architecture:**
- Input layer → Hidden layer (ReLU) → Output layer

**Usage:**
```python
from simple_mlp import SimpleMLP

model = SimpleMLP(input_size=11, hidden_size=64, output_size=1)
```

### FlexibleMLP
A configurable multi-layer perceptron with variable number of hidden layers.

**Architecture:**
- Input layer → Hidden layers (ReLU) → Output layer
- Number and size of hidden layers is configurable

**Usage:**
```python
from flexible_mlp import FlexibleMLP

# Single hidden layer
model = FlexibleMLP(input_size=11, hidden_sizes=[64], output_size=1)

# Multiple hidden layers
model = FlexibleMLP(input_size=11, hidden_sizes=[64, 32, 16], output_size=1)
```

## Running Tests

To run the model tests with pytest:

**From repository root:**
```bash
# Using virtual environment
src/dataloaders/WineQualityLoader/.venv/bin/python -m pytest src/models/tests/test_mlp.py -v
```

**Or activate the virtual environment first:**
```bash
cd src/dataloaders/WineQualityLoader
source .venv/bin/activate
cd ../../../
pytest src/models/tests/test_mlp.py -v
```

**Installing pytest (if not already installed):**
```bash
cd src/dataloaders/WineQualityLoader
uv pip install pytest
```
