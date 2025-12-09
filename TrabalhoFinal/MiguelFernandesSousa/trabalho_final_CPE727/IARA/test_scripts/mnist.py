"""
Training MNIST Test Program

This script generates a sample training configuration for MNIST dataset and traine a MLP or
RandomForest for each iara.trainer.TrainingStrategy. Can be used in balanced or unbalanced dataset.
"""
import enum
import os
import typing
import shutil
import argparse

import tqdm
import numpy as np

import torch
import torch.utils.data as torch_data
import torchvision

import iara.ml.dataset as iara_dataset
import iara.ml.models.trainer as iara_trn
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.cnn as iara_cnn
import iara.ml.metrics as iara_metrics

class Dataset(iara_dataset.BaseDataset):
    """Simple adapter for MNIST dataset keeping interface for training."""

    def __init__(self, dataset: torch_data.Dataset, indexes: typing.List[int] = None) -> None:

        self.samples = dataset.data
        self.targets = dataset.targets

        if indexes is not None:
            self.samples = self.samples[indexes]
            self.targets = self.targets[indexes]

        transform = torchvision.transforms.Normalize((0.5,), (0.5,))

        self.samples = self.samples.float()/255
        self.samples = transform(self.samples.unsqueeze(1))

    def get_targets(self) -> torch.tensor:
        return self.targets

    def get_samples(self) -> torch.tensor:
        return self.samples

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index], self.targets[index]


class Types(enum.Enum):
    """Model types for training in this test script."""
    MLP=0
    RANDOM_FOREST=1
    CNN=2

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

def main(override: bool, unbalanced: bool, training_type: Types):
    """Main function for the MNIST Test."""

    output_dir = f"./results/trainings/mnist/{str(training_type)}/{'un' if unbalanced else ''}balanced"
    model_dir = os.path.join(output_dir, 'model')
    eval_dir = os.path.join(output_dir, 'eval')

    if os.path.exists(output_dir) and override:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trn_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    if unbalanced:

        qtys = [15, 20, 50, 100, 800, 800, 800, 800, 800, 800]

        trn_indexes = []
        val_indexes = []
        for i, qty in enumerate(qtys):
            class_index = np.where(np.array(trn_dataset.targets) == i)[0]
            trn_indexes.extend(class_index[:qty])

            class_index = np.where(np.array(val_dataset.targets) == i)[0]
            val_indexes.extend(class_index[:qty])

    else:
        trn_indexes = None
        val_indexes = None


    trn_dataset = Dataset(trn_dataset, trn_indexes)
    val_dataset = Dataset(val_dataset, val_indexes)

    trainer_list = []

    if training_type == Types.MLP:

        trainer_list.append(iara_trn.OptimizerTrainer(
                    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                    trainer_id='MLP',
                    n_targets=10,
                    model_allocator=lambda input_shape, n_targets:
                            iara_mlp.MLP(
                                    input_shape=input_shape,
                                    n_neurons=32,
                                    n_targets=n_targets),
                    batch_size=64,
                    n_epochs=32,
                    patience=5))

    elif training_type == Types.RANDOM_FOREST:

        trainer_list.append(iara_trn.RandomForestTrainer(
                    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                    trainer_id = 'Forest',
                    n_targets=10,
                    n_estimators=100,
                    max_depth=None))

        # trainer_list.append(iara_trn.RandomForestTrainer(
        #             training_strategy=iara_trn.ModelTrainingStrategy.CLASS_SPECIALIST,
        #             trainer_id = 'Forest',
        #             n_targets=10,
        #             n_estimators=100,
        #             max_depth=None))

    elif training_type == Types.CNN:

        trainer_list.append(iara_trn.OptimizerTrainer(
                    training_strategy=iara_trn.ModelTrainingStrategy.MULTICLASS,
                    trainer_id='CNN',
                    n_targets=10,
                    model_allocator=lambda input_shape, n_targets:
                            iara_cnn.CNN(
                                    input_shape=input_shape,
                                    conv_activation = torch.nn.ReLU(),
                                    conv_n_neurons=[32, 64],
                                    kernel_size=3,
                                    padding=1,
                                    classification_n_neurons=128,
                                    n_targets=n_targets),
                    batch_size=64,
                    n_epochs=32,
                    patience=5))

    else:
        raise NotImplementedError('This training type its not implemented')

    grid = iara_metrics.GridCompiler()

    for trainer in tqdm.tqdm(trainer_list, leave=False, desc="Trainings"):

        trainer.fit(model_dir, trn_dataset, val_dataset)

        result = trainer.eval(dataset_id='val',
                            model_base_dir=model_dir,
                            eval_base_dir=eval_dir,
                            dataset=val_dataset)

        grid.add(params={'trainer': str(trainer)},
            i_fold=0,
            prediction=result["Prediction"],
            target=result["Target"])

    print('############################')
    print(grid)

if __name__ == "__main__":
    type_str_list = [str(i) for i in Types]

    parser = argparse.ArgumentParser(description='RUN CPA analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--unbalanced', action='store_true', default=False,
                        help='Use unbalanced training data')
    parser.add_argument('--training_type', type=str, choices=type_str_list,
                        default=None, help='Type of model for training')

    args = parser.parse_args()

    if args.training_type is None:
        for model_type in tqdm.tqdm(Types, leave=False, desc="Model Types"):
            main(args.override, args.unbalanced, model_type)
    else:
        index = type_str_list.index(args.training_type)
        main(args.override, args.unbalanced, Types(index))
