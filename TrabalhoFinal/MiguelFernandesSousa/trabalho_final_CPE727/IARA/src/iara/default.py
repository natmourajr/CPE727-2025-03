import enum
import typing

import numpy as np
import pandas as pd

import torch

import iara.records
import iara.ml.dataset as iara_dataset
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager
import iara.ml.models.trainer as iara_trn
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.models.mlp as iara_mlp

class Directories:
    """A structure for configuring directories for locating and storing files."""
    def __init__(self,
                 data_dir="./data/iara",
                 process_dir="./data/iara_processed",
                 training_dir="./results/trainings",
                 comparison_dir="./results/comparisons",
                 tables_dir="./results/tables"):
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.training_dir = training_dir
        self.comparison_dir = comparison_dir
        self.tables_dir = tables_dir


DEFAULT_DIRECTORIES = Directories()
DEFAULT_DEEPSHIP_DIRECTORIES = Directories(data_dir="/data/deepship",
                                           process_dir="./data/deepship_processed")

class Target(enum.Enum):
    # https://www.mdpi.com/2072-4292/11/3/353
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    BACKGROUND = 3

    @staticmethod
    def classify(ship_length: float) -> 'Target':
        if np.isnan(ship_length):
            return Target.BACKGROUND

        if ship_length < 50:
            return Target.SMALL
        if ship_length < 100:
            return Target.MEDIUM

        return Target.LARGE

    @staticmethod
    def classify_row(ship_length: pd.DataFrame) -> float:
        try:
            return Target.classify(float(ship_length['Length'])).value
        except ValueError:
            return np.nan

def default_iara_lofar_audio_processor(directories: Directories = DEFAULT_DIRECTORIES):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOFAR,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        integration_interval=0.512
    )

def default_iara_mel_audio_processor(directories: Directories = DEFAULT_DIRECTORIES,
                                     n_mels: int = 256):
    """Method to get default AudioFileProcessor for iara."""
    return iara_manager.AudioFileProcessor(
        data_base_dir = directories.data_dir,
        data_processed_base_dir = directories.process_dir,
        normalization = iara_proc.Normalization.NORM_L2,
        analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
        n_pts = 1024,
        n_overlap = 0,
        decimation_rate = 3,
        n_mels=n_mels,
        integration_interval=0.512
    )

def default_collection(only_sample: bool = False,
                       collection: iara.records.Collection = iara.records.Collection.OS):
    """Method to get default collection for iara."""
    if collection in [iara.records.Collection.OS, iara.records.Collection.GLIDER]:
        n_targets = 4
    else:
        n_targets = 3

    return iara.records.CustomCollection(
            collection = collection,
            target = iara.records.GenericTarget(
                n_targets = n_targets,
                function = Target.classify_row,
                include_others = False
            ),
            only_sample=only_sample
        )

def default_window_input():
    return iara_dataset.InputType.Window()

def default_image_input():
    return iara_dataset.InputType.Image(n_windows=32, overlap=0.5)


class Classifier(enum.Enum):
    FOREST = 0
    MLP = 1
    CNN = 2

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def get_input_type(self):
        if self == Classifier.CNN:
            return default_image_input()

        return default_window_input()


def default_mel_managers(config_name: str,
                         output_base_dir: str,
                         classifiers: typing.List[Classifier],
                         collection: iara.records.CustomCollection,
                         data_processor: iara_manager.AudioFileProcessor,
                         training_strategy: iara_trn.ModelTrainingStrategy = iara_trn.ModelTrainingStrategy.MULTICLASS):

    manager_dict = {}

    for classifier in classifiers:

        input = classifier.get_input_type()

        config = iara_exp.Config(
                        name = f'{config_name}_{input.type_str()}',
                        dataset = collection,
                        dataset_processor = data_processor,
                        output_base_dir = output_base_dir,
                        input_type = input)
    
        if classifier == Classifier.CNN:

            trainer = iara_trn.OptimizerTrainer(
                    training_strategy=training_strategy,
                    trainer_id = 'cnn mel',
                    n_targets = config.dataset.target.get_n_targets(),
                    batch_size = 32,
                    model_allocator = lambda input_shape, n_targets,
                        conv_n_neurons = [1024, 128],
                        conv_activation = torch.nn.LeakyReLU,
                        conv_pooling = torch.nn.MaxPool2d,
                        conv_pooling_size = [2,2],
                        conv_dropout = 0.2,
                        batch_norm = torch.nn.BatchNorm2d,
                        kernel_size = 5,

                        classification_n_neurons = 128,
                        classification_dropout = 0.4,
                        classification_norm = None,
                        classification_hidden_activation = torch.nn.ReLU,
                        classification_output_activation = torch.nn.Sigmoid:

                            iara_cnn.CNN(
                                    input_shape = input_shape,

                                    conv_n_neurons = conv_n_neurons,
                                    conv_activation = conv_activation,
                                    conv_pooling = conv_pooling,
                                    conv_pooling_size = conv_pooling_size,
                                    conv_dropout = conv_dropout,
                                    batch_norm = batch_norm,
                                    kernel_size = kernel_size,

                                    classification_n_neurons = classification_n_neurons,
                                    n_targets = n_targets,
                                    classification_dropout = classification_dropout,
                                    classification_norm = classification_norm,
                                    classification_hidden_activation = classification_hidden_activation,
                                    classification_output_activation = classification_output_activation,
                            ),
                    optimizer_allocator=lambda model,
                        weight_decay = 1e-3,
                        lr = 1e-6:
                            torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr),
                    loss_allocator = lambda class_weights:
                            torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
                    )

        elif classifier == Classifier.FOREST:
            trainer = iara_trn.RandomForestTrainer(
                    training_strategy=training_strategy,
                    trainer_id = 'forest mel',
                    n_targets = collection.target.get_n_targets(),
                    n_estimators = 50,
                    max_depth = 30)

        elif classifier == Classifier.MLP:

            trainer = iara_trn.OptimizerTrainer(
                    training_strategy=training_strategy,
                    trainer_id = 'mlp mel',
                    n_targets = config.dataset.target.get_n_targets(),
                    batch_size = 30,
                    n_epochs = 50,
                    patience = 8,
                    model_allocator = lambda input_shape, n_targets,
                        hidden_channels = [32, 16],
                        dropout = 0.2,
                        norm_layer = torch.nn.BatchNorm1d,
                        activation_layer = torch.nn.ReLU,
                        activation_output_layer = torch.nn.Sigmoid:

                            iara_mlp.MLP(input_shape = input_shape,
                                hidden_channels = hidden_channels,
                                n_targets = n_targets,
                                dropout = dropout,
                                norm_layer = norm_layer,
                                activation_layer = activation_layer,
                                activation_output_layer = activation_output_layer),

                    optimizer_allocator=lambda model,
                        weight_decay = 1e-3,
                        lr = 1e-4:
                            torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr),
                    loss_allocator = lambda class_weights:
                            torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            )

        manager_dict[classifier] = iara_exp.Manager(config, trainer)

    return manager_dict
