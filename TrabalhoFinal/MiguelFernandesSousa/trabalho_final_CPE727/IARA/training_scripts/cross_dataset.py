import enum
import os
import pandas as pd
import itertools
import shutil
import time
import argparse
import numpy as np
import typing

import torch

import iara.records
import iara.ml.experiment as iara_exp
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics
import iara.ml.dataset as iara_dataset
import iara.ml.models.cnn as iara_cnn
import iara.ml.models.mlp as iara_mlp
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


class OtherCollections(enum.Enum):
    SHIPSEAR = 0
    DEEPSHIP = 1

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

    def _get_info_filename(self) -> str:
        return os.path.join("./training_scripts/dataset_info", f"{str(self)}.csv")

    def to_df(self, only_sample: bool = False) -> pd.DataFrame:
        df = pd.read_csv(self._get_info_filename(), na_values=[" - "])
        return df

    def get_id(self, file: str) -> int:
        if self == OtherCollections.SHIPSEAR:
            return int(file.split('_')[0])
        elif self == OtherCollections.DEEPSHIP:
            return int(file.split('.')[0])
        raise NotImplementedError(f'get_id not implemented for {self}')

    def classify_row(self, df: pd.DataFrame) -> float:
        if self == OtherCollections.SHIPSEAR:
            classes_by_length = ['B', 'C', 'A', 'D', 'E']
            try:
                target = (classes_by_length.index(df['Class']) - 1)
                if target < 0:
                    return 0
                return target
            except ValueError:
                return np.nan
            
        elif self == OtherCollections.DEEPSHIP:
            return iara_default.Target.classify_row(df)

        raise NotImplementedError(f'classify_row not implemented for {self}')

    def default_mel_managers(self,
                         output_base_dir: str,
                         classifiers: typing.List[iara_default.Classifier],
                         training_strategy: iara_trn.ModelTrainingStrategy = iara_trn.ModelTrainingStrategy.MULTICLASS):

        if self != OtherCollections.SHIPSEAR:
            raise NotImplementedError(f'default_mel_managers not implemented for {self}')

        config_name = 'shipsear'
        data_base_dir = "./data/shipsear_16e3"
        data_processed_base_dir = "./data/shipsear_processed"

        collection = iara.records.CustomCollection(
                    collection = OtherCollections.SHIPSEAR,
                    target = iara.records.GenericTarget(
                        n_targets = 4,
                        function = OtherCollections.SHIPSEAR.classify_row,
                        include_others = False
                    ),
                    only_sample=False
                )

        dataset_processor = iara_manager.AudioFileProcessor(
                data_base_dir = data_base_dir,
                data_processed_base_dir = data_processed_base_dir,
                normalization = iara_proc.Normalization.NORM_L2,
                analysis = iara_proc.SpectralAnalysis.LOG_MELGRAM,
                n_pts = 1024,
                n_overlap = 0,
                decimation_rate = 1,
                n_mels=256,
                integration_interval=0.512,
                extract_id = OtherCollections.SHIPSEAR.get_id
            )

        manager_dict = {}

        for classifier in classifiers:

            input = classifier.get_input_type()

            config = iara_exp.Config(
                            name = f'{config_name}_{input.type_str()}',
                            dataset = collection,
                            dataset_processor = dataset_processor,
                            output_base_dir = output_base_dir,
                            input_type = input)
        
            if classifier == iara_default.Classifier.CNN:

                trainer = iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = 'cnn mel',
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = 32,
                        model_allocator = lambda input_shape, n_targets,
                            conv_n_neurons = [256, 32],
                            conv_activation = torch.nn.PReLU,
                            conv_pooling = torch.nn.MaxPool2d,
                            conv_pooling_size = [2,2],
                            conv_dropout = 0,
                            batch_norm = torch.nn.BatchNorm2d,
                            kernel_size = 7,

                            classification_n_neurons = [64, 32],
                            classification_dropout = 0,
                            classification_norm = None,
                            classification_hidden_activation = torch.nn.PReLU,
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
                            lr = 1e-4:
                                torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr),
                        loss_allocator = lambda class_weights:
                                torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
                        )

            elif classifier == iara_default.Classifier.FOREST:
                trainer = iara_trn.RandomForestTrainer(
                        training_strategy=training_strategy,
                        trainer_id = 'forest mel',
                        n_targets = collection.target.get_n_targets(),
                        n_estimators = 8,
                        max_depth = 5)

            elif classifier == iara_default.Classifier.MLP:
                
                trainer = iara_trn.OptimizerTrainer(
                        training_strategy=training_strategy,
                        trainer_id = 'mlp mel',
                        n_targets = config.dataset.target.get_n_targets(),
                        batch_size = 64,
                        n_epochs = 100,
                        patience = 8,
                        model_allocator = lambda input_shape, n_targets,
                            hidden_channels = 128,
                            dropout = 0.4,
                            norm_layer = torch.nn.BatchNorm1d,
                            activation_layer = torch.nn.PReLU,
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


def main(
         training_strategy: iara_trn.ModelTrainingStrategy,
         folds: typing.List[int],
         only_eval: bool,
         override: bool):


    classifiers = [iara_default.Classifier.FOREST, iara_default.Classifier.MLP, iara_default.Classifier.CNN]
    # classifiers = [iara_default.Classifier.FOREST]
    eval_subsets = [iara_trn.Subset.TEST, iara_trn.Subset.ALL]

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/cross_dataset"
    comparison_dir = f'{output_base_dir}/comparison'
    grids_dir = f'{output_base_dir}/grids'

    os.makedirs(grids_dir, exist_ok=True)

    cross_grids = {}
    cross_incomplete = False
    for classifier in classifiers:
        for subset in eval_subsets:
            cross_grids[subset, classifier] = {
                'filename': f'{grids_dir}/{classifier}_{subset}.pkl',
                'cv': None
            }
            if os.path.exists(cross_grids[subset, classifier]['filename']):
                cross_grids[subset, classifier]['cv'] = iara_metrics.GridCompiler.load(cross_grids[subset, classifier]['filename'])
            else:
                cross_incomplete = True

    iara_name = f'iara'

    manager_dict_iara = iara_default.default_mel_managers(config_name = iara_name,
                        output_base_dir = output_base_dir,
                        classifiers = classifiers,
                        collection = iara_default.default_collection(),
                        data_processor = iara_default.default_iara_mel_audio_processor(),
                        training_strategy = training_strategy)

    manager_dict_shipsear = OtherCollections.SHIPSEAR.default_mel_managers(
                        output_base_dir = output_base_dir,
                        classifiers = classifiers,
                        training_strategy = training_strategy)

    print("############ IARA ############")
    id_listA = manager_dict_iara[classifiers[-1]].config.split_datasets()
    manager_dict_iara[classifiers[-1]].print_dataset_details(id_listA)
    print("############ Shipsear ############")
    id_listB = manager_dict_shipsear[classifiers[-1]].config.split_datasets()
    manager_dict_shipsear[classifiers[-1]].print_dataset_details(id_listB)

    if cross_incomplete:

        if not only_eval:

            print("############ Training IARA ############")
            for _, manager in manager_dict_iara.items():
                manager.run(folds = folds, override = override, without_ret = True)

            print("############ Training ShipsEar ############")
            for _, manager in manager_dict_shipsear.items():
                manager.run(folds = folds, override = override, without_ret = True)

        for eval_subsets in eval_subsets:
            for classifier in classifiers:
                comparator = iara_exp.CrossComparator(comparator_eval_dir = comparison_dir,
                                                    manager_a = manager_dict_iara[classifier],
                                                    manager_b = manager_dict_shipsear[classifier])

                cross_grids[eval_subsets, classifier]['cv'] = comparator.cross_compare(
                                        eval_strategy = iara_trn.EvalStrategy.BY_AUDIO,
                                        folds = folds,
                                        eval_subset=eval_subsets)

                cross_grids[eval_subsets, classifier]['cv'].export(cross_grids[eval_subsets, classifier]['filename'])

    print("############### Cross comparison ###############")
    for (subset, classifier), cv_dict in cross_grids.items():
        print(f"--------- {subset} - {classifier} ---------")
        print(cv_dict['cv'])


if __name__ == "__main__":
    start_time = time.time()

    strategy_choises = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN GridSearch analysis', add_help=False)
    parser.add_argument('-F', '--fold', type=str, default=None,
                        help='Specify folds to be executed. Example: 0,4-7')
    parser.add_argument('-t','--training_strategy', type=str, choices=strategy_choises,
                        default=None, help='Strategy for training the model')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='Not training, only evaluate trained models')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    strategies = []
    if args.training_strategy is not None:
        index = strategy_choises.index(args.training_strategy)
        strategies.append(iara_trn.ModelTrainingStrategy(index))
    else:
        strategies = [iara_trn.ModelTrainingStrategy.MULTICLASS]

    for strategy in strategies:
        main(training_strategy = strategy,
             folds = folds_to_execute,
             only_eval = args.only_eval,
             override = args.override)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {iara.utils.str_format_time(elapsed_time)}")
