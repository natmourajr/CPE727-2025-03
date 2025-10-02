# DVSLoader

A dataloader package for loading DVS (Dynamic Vision Sensor) datasets.

## Overview

DVSLoader provides utilities for loading and preprocessing DVS datasets for neuromorphic computing applications.

## Current Datasets

### SL-Animals-DVS
This data loader is based on the datasets implemented in a previous work [1]. The codebases can be found [here](https://github.com/ronichester).
 - **Dataset**: [`SL-Animals-DVS`](https://link.springer.com/article/10.1007/s10044-021-01011-w)
 - **Preprocessing**: Currently supports DECOLLE preprocessing only

## Future Development

- Additional dataset loaders
- Multiple preprocessing options
- Extended format support

## Usage

```python
from dataloaders.DVSLoader import sl_animals_dvs_loader
```

## Requirements

- Python >=3.13
- NumPy >=2.3.3
- PyTorch >=2.8.0
- Pandas >=2.3.3
- Tonic >=1.4.3
- [DECOLLE](https://github.com/nmi-lab/decolle-public.git)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management. uv provides dependency resolution, virtual environment management, and streamlined project setup. To install the project dependencies, make sure you have uv installed and run `uv sync` in the project directory.

## Contributing

This package is under active development. More loaders and preprocessors will be added in future versions.

## References

[1] C. R. Schechter and J. G. R. C. Gomes, "Enhancing Gesture Recognition Performance Using Optimized Event-Based Data Sample Lengths and Crops," 2024 IEEE 15th Latin America Symposium on Circuits and Systems (LASCAS), Punta del Este, Uruguay, 2024, pp. 1-5, doi: 10.1109/LASCAS60203.2024.10506133.
keywords: {Training;Visualization;Sign language;Dynamics;Crops;Focusing;Vision sensors;Supervised Learning;Event-Based Camera;Gesture Recognition;High Speed Recognition;Spiking Neural Networks;SNN;Dynamic Vision Sensors;DVS;Sign-Language},