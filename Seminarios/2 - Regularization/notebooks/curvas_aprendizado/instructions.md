# Learning Curves Regularization Notebook

Jupyter notebook for using regularization techniques related to learning curves:

- Early Stopping
- Double Descent

The main objective is to be able to see the effects of regularization based on early stopping and double descent for ML models.

## Early Stopping

This project must have an implementation of early stopping in python, which follows Goodfellow algorithm:

```latex
\begin{algorithmic}[1]
\STATE Let $n$ be the number of steps between evaluations.
\STATE Let $p$ be the ``patience,'' the number of times to observe worsening validation set error before giving up.
\STATE Let $\theta_0$ be the initial parameters.
\STATE $\theta \leftarrow \theta_0$
\STATE $i \leftarrow 0$
\STATE $j \leftarrow 0$
\STATE $v \leftarrow \infty$
\STATE $\theta^\ast \leftarrow \theta$
\STATE $i^\ast \leftarrow i$
\WHILE{$j < p$}
    \STATE Update $\theta$ by running the training algorithm for $n$ steps.
    \STATE $i \leftarrow i + n$
    \STATE $v' \leftarrow \text{ValidationSetError}(\theta)$
    \IF{$v' < v$}
        \STATE $j \leftarrow 0$
        \STATE $\theta^\ast \leftarrow \theta$
        \STATE $i^\ast \leftarrow i$
        \STATE $v \leftarrow v'$
    \ELSE
        \STATE $j \leftarrow j + 1$
    \ENDIF
\ENDWHILE
\STATE \textbf{Return} best parameters $\theta^\ast$ and best number of training steps $i^\ast$.
\end{algorithmic}
```

The validation set error calculation and the writing of the values from the output of it can be implemented making it run assynchronously, so that the training process is not interrupted.

For the neural network implementation, it must use an early stopping from some well known ML lib. then use this written. To compare if it works properly or not.

## Double Descent

It must use **ResNet-18** as the base architecture. It must train models with different width multipliers (e.g., 0.25x, 0.5x, 1x, 2x, 4x of the standard ResNet-18 channel widths): the more the complexity, the double descent phenomenon will appear.

It must plot both the test error as a function of the model width/complexity. Also, it must plot, for 3 fixed width multipliers, the test error as a function of epochs to observe the behavior during training.

## Implementation Instructions

### Codebase

First, I want it to make it work with python code. **DON'T MAKE THE NOTEBOOK FIRST.** We will work on that later.

It is easier to run and evaluate python project without jupyter notebooks.

**Framework:** Use PyTorch for all implementations.

### Dataset

Use **CIFAR-10** dataset.

**Data preprocessing:**
- Standard normalization using CIFAR-10 mean and std
- NO data augmentation strategies
- Train/validation split from training set (e.g., 80/20 or 90/10)

### Models

Implement three model architectures:
1. **MLP** - Basic multi-layer perceptron
2. **CNN** - Convolutional neural network
3. **ResNet-18** - For double descent experiments

### Hyperparameters & Technical Specifications

**NOTE:** Optimal hyperparameters are not known in advance. The code must be designed to try a **small set of combinations** to explore what works best. Implement experiments with the following ranges:

**Early Stopping Parameters:**
- `n` (steps between evaluations): Try [50, 10, 200] steps
- `p` (patience): Try [3, 5, 10] iterations
- Validation split ratio: Try [0.1, 0.2]

**Training Parameters:**
- Batch size: Try [32, 64, 128]
- Learning rate: Try [0.001, 0.01, 0.1]
- Optimizer: Start with Adam and SGD
- Max epochs: 200 (or until early stopping triggers)

**Model Architectures (to experiment):**
- MLP: Try [2, 3, 4] hidden layers with [128, 256, 512] neurons
- CNN: Try [2, 3] conv blocks with [32, 64] initial filters
- ResNet-18 widths: Try at least [0.25, 0.5, 1.0, 2.0, 4.0] multipliers

The code should log results for different combinations and help identify which configurations work best.

### Evaluation Metrics

Use metrics appropriate for multi-class image classification:
- **Accuracy** (primary metric for model selection)
- **Cross-entropy loss** (for training monitoring)
- **Top-5 accuracy** (for CIFAR-10 specifically)

### Hardware Constraints

- **No GPU available** - optimize for CPU training
- Use smaller batch sizes if needed
- Consider reducing model sizes for faster experimentation
- Implement progress bars and time estimates

### Reproducibility

- Set random seeds for reproducibility (numpy, torch, random)
- Log all hyperparameters used in each experiment
- Save configuration files for each run

### Visualization & Output

All plots must be **saved automatically** to disk:
- Create output folders using timestamps: `results/YYYYMMDD_HHMMSS/`
- Save plots as PNG files
- **DO NOT display/show plots** during execution - only save them
- Generate the following plots:
  - Training and validation loss curves
  - Training and validation accuracy curves
  - Early stopping comparison (with vs without)
  - Double descent: test error vs model complexity
  - Double descent: test error vs epochs (for 3 fixed widths)
  - Hyperparameter comparison plots

### Project Structure

Organize code with the following structure:
```
curvas_aprendizado/
├── data/              # Dataset loading and preprocessing
├── models/            # Model architectures (MLP, CNN, ResNet)
├── training/          # Training loops, early stopping implementation
├── evaluation/        # Evaluation metrics and test functions
├── visualization/     # Plotting and results visualization
├── configs/           # Configuration files for experiments
├── results/           # Generated plots and logs (timestamped folders)
├── notebooks/         # Phase 2: Jupyter notebooks
├── requirements.txt   # Python dependencies
├── .venv/            # Virtual environment (local)
└── README.md         # Project documentation
```

### Checkpointing

- Save best model weights during training
- Save checkpoints when early stopping triggers
- Store models in timestamped results folders

## Phases

**Phase 1:** Python project implementation
- Implement all models (MLP, CNN, ResNet-18)
- Implement custom early stopping following Goodfellow's algorithm
- Implement library-based early stopping for comparison
- Implement double descent experiments with ResNet-18 width variations
- Generate all required plots
- Log all experiments and results
- Code documented and reproducible

**Phase 1 Acceptance Criteria:**
- ✅ All models trained successfully on CIFAR-10
- ✅ All plots generated and saved to timestamped folders
- ✅ Results logged and reproducible with random seeds
- ✅ Code documented with docstrings and comments
- ✅ Hyperparameter exploration completed
- ✅ **Explicit confirmation from user that Phase 1 is complete**

**Phase 2:** Jupyter notebooks (starts ONLY after Phase 1 is complete)
- Create interactive notebooks based on Phase 1 code
- Add explanations and visualizations
- Document findings and insights

## Technical Instructions

- Use `requirements.txt` to hold all libraries and versions
- Create `.venv` local virtual environment
- Use Python 3.8+ compatible code
- Follow PEP 8 style guidelines
- Add comprehensive logging using Python's logging module
- Include error handling and validation
