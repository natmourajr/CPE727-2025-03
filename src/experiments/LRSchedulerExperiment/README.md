# Learning Rate Scheduler Experiment

This is an experiment with the following characteristics:

- Depends on a data loader whose source is located at `../dataloaders/TemplateLoader`. It is installed as a standard Python dependency.
- Includes a Dockerfile with CUDA support.

## Experiment Overview

This experiment trains a simple MLP model on the Breast Cancer dataset with preprocessing and supports flexible learning rate schedulers.

**Default parameters:**

- `epochs`: 100  
- `batch_size`: 32  
- `learning_rate`: 0.05  
- `hidden_size`: 64  
- `feature_strategy`: `onehot`  
- `target_strategy`: `binary`  
- `handle_missing`: `drop`  
- `device`: `cpu`  
- `scheduler_name`: `CosineAnnealingLR`  
- `scheduler_params`: `{}` (JSON string)  

**Output:**

- Train loss per epoch plot: `train_loss_per_epoch_<SCHEDULER_NAME>.png`  
- Learning rate per epoch plot: `lr_per_epoch_<SCHEDULER_NAME>.png`

## Running the Experiment

You can run the experiment using Python’s module syntax and the CLI script:

```bash
python -m src.experiments.LRSchedulerExperiment.lr_scheduler_experiment.cli \
    --scheduler <SCHEDULER_NAME> \
    --scheduler-params '<JSON_PARAMS>'
