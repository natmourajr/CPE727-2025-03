"""
Experiment Module

This module provides classes for configure, training and compare machine learning models.
"""
import os
import typing
import datetime
import itertools
import tqdm

import tqdm
import pandas as pd
import jsonpickle

import sklearn.model_selection as sk_selection

import iara.utils
import iara.records
import iara.ml.metrics as iara_metrics
import iara.ml.dataset as iara_dataset
import iara.ml.models.trainer as iara_trainer
import iara.processing.manager as iara_manager

class Config:
    """Class representing training configuration."""
    TIME_STR_FORMAT = "%Y%m%d-%H%M%S"

    def __init__(self,
                name: str,
                dataset: iara.records.CustomCollection,
                dataset_processor: iara_manager.AudioFileProcessor,
                input_type: iara_dataset.InputType,
                output_base_dir: str,
                test_ratio: float = 0.25,
                exclusive_ships_on_test = True,
                exclusive_header = 'ID',
                target_header = 'Target'):
        """
        Parameters:
        - name (str): A unique identifier for the training configuration.
        - dataset (iara.description.CustomDataset): The dataset used for training.
        - dataset_processor (iara.processing.dataset.DatasetProcessor):
            The DatasetProcessor for accessing and processing data in the dataset.
        - output_base_dir (str): The base directory for storing training outputs.
        - n_folds (int, optional): Number of folds for Kfold cross-validation. Default is 10.
        - test_factor (float, optional): Fraction of the dataset reserved for the test subset.
            Default is 0.2 (20%).
        """
        self.name = name
        self.dataset = dataset
        self.dataset_processor = dataset_processor
        self.input_type = input_type
        self.output_base_dir = os.path.join(output_base_dir, self.name)
        self.test_ratio = test_ratio
        self.exclusive_ships_on_test = exclusive_ships_on_test
        self.exclusive_header = exclusive_header
        self.target_header = target_header

    def get_n_folds(self) -> int:
        return 10

    def __str__(self) -> str:
        return  f"----------- {self.name} ----------- \n{str(self.dataset)}"

    def save(self, file_dir: str) -> None:
        """Save the Config to a JSON file."""
        os.makedirs(file_dir, exist_ok = True)
        file_path = os.path.join(file_dir, f"{self.name}.json")
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json_str = jsonpickle.encode(self, indent=4)
            json_file.write(json_str)

    @staticmethod
    def load(file_dir: str, name: str) -> 'Config':
        """Read a JSON file and return a Config instance."""
        file_path = os.path.join(file_dir, f"{name}.json")
        with open(file_path, 'r', encoding="utf-8") as json_file:
            json_str = json_file.read()
        return jsonpickle.decode(json_str)

    def split_datasets(self) -> \
        typing.List[typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split the dataset into training, validation, and test sets. 5x2 cv

        Returns:
        - list (list): List of tuples containing training, validation and test sets
            for each fold in StratifiedKFold strategy
        """
        df = self.dataset.to_df()

        if self.exclusive_header in df.columns:
            df_ships = df[~df[self.exclusive_header].isna()]
            df_non_ships = df[df[self.exclusive_header].isna()]

            df_filtered = df_ships.drop_duplicates(subset=self.exclusive_header)
        else:
            df_filtered = pd.DataFrame()
            df_non_ships = df

        split_list = []

        for i in range(5):

            skf = sk_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=42*i)

            if not df_filtered.empty:
                ship_split = skf.split(df_filtered, df_filtered[self.target_header])
            else:
                ship_split = [None, None]

            if not df_non_ships.empty:
                non_ship_split = skf.split(df_non_ships, df_non_ships[self.target_header])
            else:
                non_ship_split = [None, None]

            for (ship_idx, non_ship_idx) in zip(ship_split, non_ship_split):

                sss = sk_selection.StratifiedShuffleSplit(n_splits=1,
                                                        test_size=self.test_ratio,
                                                        random_state=42*i)

                if ship_idx is not None:
                    train_ship_ids = df_filtered.iloc[ship_idx[0]][self.exclusive_header]
                    val_test_ship_ids = df_filtered.iloc[ship_idx[1]]

                    (test_idx, val_idx) = next(sss.split(val_test_ship_ids, val_test_ship_ids[self.target_header]))

                    val_ship_ids = val_test_ship_ids.iloc[val_idx][self.exclusive_header]
                    test_ship_ids = val_test_ship_ids.iloc[test_idx][self.exclusive_header]

                    train_data = df_ships[df_ships[self.exclusive_header].isin(train_ship_ids)]
                    val_data = df_ships[df_ships[self.exclusive_header].isin(val_ship_ids)]
                    test_data = df_ships[df_ships[self.exclusive_header].isin(test_ship_ids)]


                if non_ship_idx is not None:

                    val_test_non_ship_ids = df_non_ships.iloc[non_ship_idx[1]]

                    (test_idx, val_idx) = next(sss.split(val_test_non_ship_ids, val_test_non_ship_ids[self.target_header]))

                    if ship_idx is not None:
                        train_data = pd.concat([train_data, df_non_ships.iloc[non_ship_idx[0]]])
                        val_data = pd.concat([val_data, val_test_non_ship_ids.iloc[val_idx]])
                        test_data = pd.concat([test_data, val_test_non_ship_ids.iloc[test_idx]])
                    else:
                        train_data = df_non_ships.iloc[non_ship_idx[0]]
                        val_data = val_test_non_ship_ids.iloc[val_idx]
                        test_data = val_test_non_ship_ids.iloc[test_idx]

                split_list.append((train_data, val_data, test_data))

        return split_list

    def get_data_loader(self) -> iara_dataset.ExperimentDataLoader:
        df = self.dataset.to_df()

        if 'CPA time' in df.columns:
            return iara_dataset.ExperimentDataLoader(self.dataset_processor,
                                            df['ID'].to_list(),
                                            df['Target'].to_list(),
                                            df['CPA time'].to_list())

        return iara_dataset.ExperimentDataLoader(self.dataset_processor,
                                        df['ID'].to_list(),
                                        df['Target'].to_list())

    def __eq__(self, other):
        if isinstance(other, Config):
            return (self.name == other.name and
                    # self.dataset == other.dataset and
                    self.dataset_processor == other.dataset_processor and
                    self.input_type == other.input_type and
                    self.output_base_dir == other.output_base_dir and
                    self.test_ratio == other.test_ratio and
                    self.exclusive_ships_on_test == other.exclusive_ships_on_test)
        return False


class Manager():
    """Class for managing and executing training based on a Config for multiple trainers"""

    def __init__(self,
                 config: Config,
                 *trainers: iara_trainer.BaseTrainer) -> None:
        """
        Args:
            config (Config): The training configuration.
            trainer_list (typing.List[BaseTrainer]): A list of BaseTrainer instances to be used
                for training and evaluation in this configuration.
        """
        self.config = config
        self.trainer_list = trainers
        self.experiment_loader = None

    def get_experiment_loader(self) -> iara_dataset.ExperimentDataLoader:
        if self.experiment_loader is None:
            self.experiment_loader = self.config.get_data_loader()
        return self.experiment_loader

    def __str__(self) -> str:
        return f'{self.config.name} with {len(self.trainer_list)} models'

    def __prepare_output_dir(self, override: bool):
        """ Creates the directory tree for training, keeping backups of conflicting trainings. """
        if os.path.exists(self.config.output_base_dir):
            try:
                if not override:
                    old_config = Config.load(self.config.output_base_dir, self.config.name)

                    if old_config == self.config:
                        return

                iara.utils.backup_folder(base_dir=self.config.output_base_dir,
                                         time_str_format=Config.TIME_STR_FORMAT)

            except FileNotFoundError:
                pass

        os.makedirs(self.config.output_base_dir, exist_ok=True)
        self.config.save(self.config.output_base_dir)

    def get_model_base_dir(self, i_fold: int) -> str:
        return os.path.join(self.config.output_base_dir,
                                'model',
                                f'fold_{i_fold}')

    def is_trained(self, i_fold: int) -> bool:
        model_base_dir = self.get_model_base_dir(i_fold)
        for trainer in self.trainer_list:
            if not trainer.is_trained(model_base_dir):
                return False
        return True

    def fit(self, i_fold: int,
            trn_dataset_ids: typing.Iterable[int],
            val_dataset_ids: typing.Iterable[int]) -> None:
        """
        Fit the model for a specified fold using the provided training and validation dataset IDs
            and targets.

        Args:
            i_fold (int): The index of the fold.
            trn_dataset_ids (typing.Iterable[int]): Iterable of training dataset IDs.
            val_dataset_ids (typing.Iterable[int]): Iterable of validation dataset IDs.
        """
        if self.is_trained(i_fold):
            return

        trn_dataset = iara_dataset.AudioDataset(self.get_experiment_loader(), self.config.input_type, trn_dataset_ids)

        val_dataset = iara_dataset.AudioDataset(self.get_experiment_loader(), self.config.input_type, val_dataset_ids)

        model_base_dir = self.get_model_base_dir(i_fold)

        for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                        tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers", ncols=120):

            iara.utils.set_seed()

            trainer.fit(model_base_dir=model_base_dir,
                    trn_dataset=trn_dataset,
                    val_dataset=val_dataset)

    def is_evaluated(self,
            eval_subset: iara_trainer.Subset,
            eval_base_dir: str) -> bool:
        for trainer in self.trainer_list:
            if not trainer.is_evaluated(eval_subset = eval_subset,
                                        eval_base_dir = eval_base_dir):
                return False
        return True

    def eval(self,
             i_fold: int,
             eval_subset: iara_trainer.Subset,
             eval_strategy: iara_trainer.EvalStrategy,
             dataset_ids: typing.Iterable[int]) -> None:

        model_base_dir = self.get_model_base_dir(i_fold)
        eval_base_dir = os.path.join(self.config.output_base_dir,
                                        'eval',
                                        f'fold_{i_fold}')

        if not self.is_trained(i_fold):
            raise FileNotFoundError(f'Models not trained in {model_base_dir}')

        if self.is_evaluated(eval_subset = eval_subset,
                             eval_base_dir=eval_base_dir):
            return

        dataset = iara_dataset.AudioDataset(
                    self.get_experiment_loader(),
                    self.config.input_type,
                    dataset_ids)

        for trainer in self.trainer_list if (len(self.trainer_list) == 1) else \
                            tqdm.tqdm(self.trainer_list, leave=False, desc="Trainers", ncols=120):

            trainer.eval(eval_subset = eval_subset,
                         eval_strategy = eval_strategy,
                         model_base_dir=model_base_dir,
                         eval_base_dir=eval_base_dir,
                         dataset=dataset)

    def compile_results(self,
                        eval_subset: iara_trainer.Subset,
                        eval_strategy: iara_trainer.EvalStrategy,
                        trainer_list: typing.List[iara_trainer.BaseTrainer] = None,
                        folds: typing.List[int] = None) -> typing.List[pd.DataFrame]:

        result_dict = {}

        for trainer in trainer_list if trainer_list is not None else self.trainer_list:

            results = []
            for i_fold in range(self.config.get_n_folds()):

                if folds and i_fold not in folds:
                    continue

                eval_base_dir = os.path.join(self.config.output_base_dir,
                                                'eval',
                                                f'fold_{i_fold}')

                results.append(trainer.eval(eval_subset=eval_subset,
                                            eval_strategy=eval_strategy,
                                            eval_base_dir=eval_base_dir))

            result_dict[trainer.trainer_id] = results

        return result_dict

    def compile_existing_results(self,
                        eval_subset: iara_trainer.Subset,
                        eval_strategy: iara_trainer.EvalStrategy,
                        trainer_list: typing.List[iara_trainer.BaseTrainer] = None,
                        only_eval_completed: bool = False,
                        folds: typing.List[int] = None) -> typing.List[pd.DataFrame]:

        result_dict = {}

        for trainer in tqdm.tqdm(trainer_list if trainer_list is not None else self.trainer_list,
                                    leave=False,
                                    desc="Trainers",
                                    ncols=120):

            results = []
            for i_fold in folds:

                eval_base_dir = os.path.join(self.config.output_base_dir,
                                                'eval',
                                                f'fold_{i_fold}')
                
                if not trainer.is_evaluated(eval_subset=eval_subset,
                                            eval_base_dir=eval_base_dir):
                    continue

                results.append(trainer.eval(eval_subset=eval_subset,
                                            eval_strategy=eval_strategy,
                                            eval_base_dir=eval_base_dir))

            if (len(results) != len(folds)) and only_eval_completed:
                continue

            result_dict[trainer.trainer_id] = results

        return result_dict

    def print_dataset_details(self, id_list) -> None:

        df = self.config.dataset.to_compiled_df()
        df = df.rename(columns={'Qty': 'Total'})

        for i_fold, (trn_set, val_set, test_set) in enumerate(id_list):

            df_trn = self.config.dataset.to_compiled_df(trn_set)
            df_val = self.config.dataset.to_compiled_df(val_set)
            df_test = self.config.dataset.to_compiled_df(test_set)

            df_trn = df_trn.rename(columns={'Qty': f'Trn_{i_fold}'})
            df_val = df_val.rename(columns={'Qty': f'Val_{i_fold}'})
            df_test = df_test.rename(columns={'Qty': f'Test_{i_fold}'})

            df = pd.merge(df, df_trn, on=self.config.dataset.target.grouped_column())
            df = pd.merge(df, df_val, on=self.config.dataset.target.grouped_column())
            df = pd.merge(df, df_test, on=self.config.dataset.target.grouped_column())

            break

        print(f'--- Dataset with {len(id_list)} n_folds ---')
        print(df)

    def run(self, folds: typing.List[int] = range, override: bool = False, without_ret = False) -> typing.Dict:
        """Execute training based on the Config"""
        self.__prepare_output_dir(override=override)
        id_list = self.config.split_datasets()

        self.print_dataset_details(id_list)

        df = self.config.dataset.to_df()

        for i_fold in folds if len(folds) == 1 else \
                                tqdm.tqdm(folds,
                                            leave=False,
                                            desc="Fold",
                                            ncols=120):
            (trn_set, val_set, test_set) = id_list[i_fold]


            for _ in tqdm.tqdm(range(1),
                               leave=False,
                                desc="--- Fitting ---",
                                bar_format = "{desc}"):

                self.fit(i_fold=i_fold,
                        trn_dataset_ids=trn_set['ID'].to_list(),
                        val_dataset_ids=val_set['ID'].to_list())

            id_set = {
                iara_trainer.Subset.TRN: trn_set,
                iara_trainer.Subset.VAL: val_set,
                iara_trainer.Subset.TEST: test_set,
                iara_trainer.Subset.ALL: df
            }

            for _ in tqdm.tqdm(range(1),
                               leave=False,
                                desc="--- Evaluating ---",
                                bar_format = "{desc}"):

                for eval_subset, eval_strategy in \
                        tqdm.tqdm(itertools.product(iara_trainer.Subset, iara_trainer.EvalStrategy),
                                leave=False,
                                desc="Subsets",
                                ncols=120):

                    self.eval(i_fold=i_fold,
                                eval_subset=eval_subset,
                                eval_strategy=eval_strategy,
                                dataset_ids=id_set[eval_subset]['ID'].to_list())

        if without_ret:
            return None

        eval_dict = {}
        for eval_subset, eval_strategy in \
                tqdm.tqdm(itertools.product(iara_trainer.Subset, iara_trainer.EvalStrategy),
                        leave=False,
                        desc="Compiling results",
                        ncols=120):

            eval_dict[(eval_subset, eval_strategy)] = \
                    self.compile_results(eval_subset=eval_subset,
                                        eval_strategy=eval_strategy,
                                        trainer_list=self.trainer_list,
                                        folds=folds)

        return eval_dict


class CrossComparator():

    def __init__(self, comparator_eval_dir: str, manager_a: Manager, manager_b: Manager) -> None:
        self.comparator_eval_dir = comparator_eval_dir
        self.manager_a = manager_a
        self.manager_b = manager_b

    def __eval_trainer(self,
                       grid: iara_metrics.GridCompiler,
                       trainer: iara_trainer.BaseTrainer,
                       manager_a: Manager,
                       manager_b: Manager,
                       eval_strategy: iara_trainer.EvalStrategy,
                       eval_subset: iara_trainer.Subset = iara_trainer.Subset.TEST,
                       folds: typing.List[int] = None):

        df = manager_b.config.dataset.to_df()
        id_list = manager_b.config.split_datasets()
        loader = manager_b.get_experiment_loader()
        
        for i_fold in folds if len(folds) == 1 else tqdm.tqdm(folds,
                                                                leave=False,
                                                                desc="Fold",
                                                                ncols=120):

            eval_base_dir = os.path.join(manager_a.config.output_base_dir,
                                            'eval',
                                            f'fold_{i_fold}')
            evaluation = trainer.eval(eval_subset=eval_subset,
                                eval_strategy=eval_strategy,
                                eval_base_dir=eval_base_dir)

            param_dict = {
                'Trainer': trainer.trainer_id,
                'Trained': manager_a.config.name,
                'Evaluated': manager_a.config.name
            }

            grid.add(params=param_dict,
                        i_fold=i_fold,
                        target=evaluation['Target'],
                        prediction=evaluation['Prediction'])


            (trn_set, val_set, test_set) = id_list[i_fold]

            if eval_subset == iara_trainer.Subset.TRN:
                ids = trn_set['ID'].to_list()
            elif eval_subset == iara_trainer.Subset.VAL:
                ids = val_set['ID'].to_list()
            elif eval_subset == iara_trainer.Subset.TEST:
                ids = test_set['ID'].to_list()
            elif eval_subset == iara_trainer.Subset.ALL:
                ids = df['ID'].to_list()

            loader.pre_load(ids)

            dataset = iara_dataset.AudioDataset(
                        loader,
                        manager_b.config.input_type,
                        ids)

            model_base_dir = manager_a.get_model_base_dir(i_fold)

            evaluation = trainer.eval(eval_subset = eval_subset,
                                        eval_strategy = eval_strategy,
                                        eval_base_dir = f'{self.comparator_eval_dir}/{manager_a.config.name}',
                                        model_base_dir = model_base_dir,
                                        dataset = dataset,
                                        complement_id=str(i_fold))

            param_dict = {
                'Trainer': trainer.trainer_id,
                'Trained': manager_a.config.name,
                'Evaluated': manager_b.config.name
            }

            grid.add(params=param_dict,
                        i_fold=i_fold,
                        target=evaluation['Target'],
                        prediction=evaluation['Prediction'])

    def cross_compare(self,
                      eval_strategy: iara_trainer.EvalStrategy,
                      eval_subset: iara_trainer.Subset = iara_trainer.Subset.TEST,
                      folds: typing.List[int] = None):

        grid = iara_metrics.GridCompiler()

        for trainer in tqdm.tqdm(self.manager_a.trainer_list,
                                    leave=False,
                                    desc="Trainers",
                                    ncols=120):

            self.__eval_trainer(
                       grid = grid,
                       trainer = trainer,
                       manager_a = self.manager_a,
                       manager_b = self.manager_b,
                       eval_strategy = eval_strategy,
                       eval_subset = eval_subset,
                       folds = folds)


        for trainer in tqdm.tqdm(self.manager_b.trainer_list,
                                    leave=False,
                                    desc="Trainers",
                                    ncols=120):

            self.__eval_trainer(
                       grid = grid,
                       trainer = trainer,
                       manager_a = self.manager_b,
                       manager_b = self.manager_a,
                       eval_strategy = eval_strategy,
                       eval_subset = eval_subset,
                       folds = folds)

        return grid
    
    def __eval_dataset(self,
                       grid: iara_metrics.GridCompiler,
                       manager: Manager,
                       trainer: iara_trainer.BaseTrainer,
                       dataset: iara_dataset.AudioDataset,
                       eval_strategy: iara_trainer.EvalStrategy = iara_trainer.EvalStrategy.BY_AUDIO,
                       eval_subset: iara_trainer.Subset = iara_trainer.Subset.ALL,
                       folds: typing.List[int] = None):

        for i_fold in folds:

            model_base_dir = manager.get_model_base_dir(i_fold)

            evaluation = trainer.eval(eval_subset = eval_subset,
                                        eval_strategy = eval_strategy,
                                        eval_base_dir = f'{self.comparator_eval_dir}/{manager.config.name}',
                                        model_base_dir = model_base_dir,
                                        dataset = dataset,
                                        complement_id=str(i_fold))

            param_dict = {
                'Trainer': trainer.trainer_id,
                'Trained': manager.config.name
            }

            grid.add(params=param_dict,
                        i_fold=i_fold,
                        target=evaluation['Target'],
                        prediction=evaluation['Prediction'])

    def cross_compare_outsource(self,
                dataset: iara_dataset.AudioDataset,
                eval_strategy: iara_trainer.EvalStrategy = iara_trainer.EvalStrategy.BY_AUDIO,
                eval_subset: iara_trainer.Subset = iara_trainer.Subset.ALL,
                folds: typing.List[int] = None):

        grid = iara_metrics.GridCompiler()

        for trainer in tqdm.tqdm(self.manager_a.trainer_list,
                                    leave=False,
                                    desc="Trainers",
                                    ncols=120):

            self.__eval_dataset(grid = grid,
                    manager = self.manager_a,
                    trainer = trainer,
                    dataset = dataset,
                    eval_strategy = eval_strategy,
                    eval_subset = eval_subset,
                    folds = folds)

        for trainer in tqdm.tqdm(self.manager_b.trainer_list,
                                    leave=False,
                                    desc="Trainers",
                                    ncols=120):

            self.__eval_dataset(grid = grid,
                    manager = self.manager_b,
                    trainer = trainer,
                    dataset = dataset,
                    eval_strategy = eval_strategy,
                    eval_subset = eval_subset,
                    folds = folds)

        return grid