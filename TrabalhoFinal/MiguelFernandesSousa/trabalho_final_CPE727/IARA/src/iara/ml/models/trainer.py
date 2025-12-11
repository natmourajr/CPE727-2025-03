"""
Trainer description Module

This module provides classes for training machine learning models.
"""
import os
import enum
import typing
import abc
import collections
import pickle

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.utils.class_weight as sk_utils

import torch
import torch.utils.data as torch_data

import iara.utils
import iara.records
import iara.ml.models.forest as iara_forest
import iara.ml.models.base_model as iara_model
import iara.ml.dataset as iara_dataset


class ModelTrainingStrategy(enum.Enum):
    """Enum defining model training strategies."""
    CLASS_SPECIALIST =0
    MULTICLASS=1

    def default_loss(self, class_weights: torch.Tensor) -> torch.nn.Module:
        """ This method returns the default loss function corresponding to the specified
                TrainingStrategy.

        Args:
            class_weights (torch.Tensor): Class weights for the loss function.

        Raises:
            NotImplementedError: If some features are still in development.

        Returns:
            torch.nn.Module: The torch loss function module.
        """
        if self == ModelTrainingStrategy.CLASS_SPECIALIST:
            return torch.nn.BCELoss(weight=class_weights[1]/class_weights[0],
                                    reduction='mean')

        if self == ModelTrainingStrategy.MULTICLASS:
            return torch.nn.CrossEntropyLoss(weight=class_weights,
                                             reduction='mean')

        raise NotImplementedError('TrainingStrategy has not default_loss implemented')

    def to_str(self, target_id: typing.Optional[int] = None) -> str:
        """This method returns a string representation that can be used to identify the model.

        Args:
            target_id typing.Optional[int]: The class identification when the model is specialized
                for a particular class. Defaults to None.

        Raises:
            NotImplementedError: If some features are still in development.

        Returns:
            str: A string to use as the model identifier.
        """
        if self == ModelTrainingStrategy.CLASS_SPECIALIST:
            if target_id is None:
                return "specialist"
            return f"specialist({target_id})"

        if self == ModelTrainingStrategy.MULTICLASS:
            return "multiclass"

        raise NotImplementedError('TrainingStrategy has not default_loss implemented')

    def __str__(self) -> str:
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

class Subset(enum.Enum):
    TRN = 0
    TRAIN = 0
    VAL = 1
    VALIDATION = 1
    TEST = 2
    ALL = 3

    def __str__(self) -> str:
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

class EvalStrategy(enum.Enum):
    BY_WINDOW = 0
    BY_AUDIO = 1

    def __str__(self) -> str:
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

class BaseTrainer():
    """
    Abstract base class defining methods that should be implemented by subclasses
    to serve as a trainer for models in the experiment module.

    This class defines the interface for trainers that are responsible for training
    and evaluating models during experiments. Subclasses must implement the required
    methods to customize the training behavior for specific models.
    """

    def __init__(self,
                 training_strategy: ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int) -> None:
        self.training_strategy = training_strategy
        self.trainer_id = trainer_id
        self.n_targets = n_targets

    def __str__(self) -> str:
        return f'{self.trainer_id}_{str(self.training_strategy)}'

    @staticmethod
    def _class_weight(dataset: iara_dataset.BaseDataset, target_id: int = None) -> torch.Tensor:
        targets = dataset.get_targets()
        if target_id is not None:
            targets = torch.where(targets == target_id, torch.tensor(1.0), torch.tensor(0.0))

        targets = targets.numpy()
        classes=np.unique(targets)

        class_weights = sk_utils.compute_class_weight('balanced',
                                                    classes=classes,
                                                    y=targets)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def output_filename(self,
                    model_base_dir: str,
                    target_id: typing.Optional[int] = None,
                    complement: str = None,
                    extention: str = 'pkl') -> str:
        """
        Return the standard model name based on the training strategy, trainer ID, and parameters.

        Args:
            target_id (typing.Optional[int], optional): The class identification when the model is
                specialized for a particular class. Defaults to None.
            extension (str, optional): The output file extension. Defaults to '.pkl'.

        Returns:
            str: The standard model name based on the provided parameters.
        """
        sufix = self.training_strategy.to_str(target_id=target_id)
        if complement is not None:
            sufix = f"{sufix}_{complement}"
        return os.path.join(model_base_dir, f'{str(self.trainer_id)}_{sufix}.{extention}')

    @abc.abstractmethod
    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        """
        Abstract method to fit (train) the model.

        This method should be implemented by subclasses to fit (train) the model using the provided
            training and validation datasets.

        Args:
            model_base_dir (str): The base directory to save any training-related outputs
                or artifacts.
            trn_dataset (iara_dataset.BaseDataset): The training dataset.
            val_dataset (iara_dataset.BaseDataset): The validation dataset.

        Returns:
            None
        """

    def is_trained(self, model_base_dir: str) -> bool:
        """
        Check if all models are trained and saved in the specified directory.

        Args:
            model_base_dir (str): The directory where the trained models are expected to be saved.

        Returns:
            bool: True if all training is completed and models are saved in the directory,
                False otherwise.
        """
        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            filename = self.output_filename(model_base_dir=model_base_dir)
            return os.path.exists(filename)

        if self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
            for target_id in range(self.n_targets):
                filename = self.output_filename(model_base_dir=model_base_dir, target_id=target_id)
                if not os.path.exists(filename):
                    return False

            return True

        raise NotImplementedError(f'TrainingStrategy has not is_trained implemented for \
                                  {self.training_strategy}')

    def predict(self,
                model: typing.Union[iara_model.BaseModel, typing.List[iara_model.BaseModel]],
                samples: torch.Tensor) -> torch.Tensor:

        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            predictions = model(samples)

        # elif self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
            # predictions = torch.zeros((len(samples), len(model)))
            # for m_idx, m in enumerate(model):
            #     predictions[:, m_idx] = m(samples)

            # predictions = predictions.argmax(dim=1)

        else:
            raise NotImplementedError(f'TrainingStrategy has not predict implemented for \
                                    {self.training_strategy}')

        return predictions

    def load(self, model_base_dir: str) -> iara_model.BaseModel:

        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            filename = self.output_filename(model_base_dir=model_base_dir)
            if not os.path.exists(filename):
                raise FileNotFoundError(f"The model file '{filename}' does not exist. Ensure \
                                        that the model is trained before evaluating.")

            model = iara_model.BaseModel.load(filename)

        # elif self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
        #     models = []
        #     for target_id in range(self.n_targets):
        #         filename = self.output_filename(model_base_dir=model_base_dir,
        #                                         target_id=target_id)
        #         if not os.path.exists(filename):
        #             raise FileNotFoundError(f"The model file '{filename}' does not exist. \
        #                                     Ensure that the model is trained before \
        #                                     evaluating.")

        #         model = iara_model.BaseModel.load(filename)
        #         models.append(model)

        else:
            raise NotImplementedError(f'TrainingStrategy has not load implemented for \
                                    {self.training_strategy}')
        
        return model

    @abc.abstractmethod
    def eval(self,
            eval_subset: Subset,
            eval_strategy: EvalStrategy,
            eval_base_dir: str,
            model_base_dir: typing.Optional[str] = None,
            dataset: iara_dataset.BaseDataset = None,
            complement_id: str = None) -> pd.DataFrame:
        """
        Abstract method to evaluate the model.

        This method should be implemented by subclasses to evaluate a trained model using the
            provided dataset.

        Args:
            dataset_id (str): Identifier for the evaluation.
            eval_base_dir (str): The base directory to save any evaluation-related outputs
                or artifacts.
            dataset typing.Optional(iara_dataset.BaseDataset): The dataset to evaluate with non set
                the evaluation must already be done.

        Returns:
            pandas.DataFrame: DataFrame with two columns, ["Target", "Prediction"]
        """
        with torch.no_grad():

            output_file = self.output_filename(model_base_dir=eval_base_dir,
                                        complement=f'{str(eval_subset)}_{complement_id}' if complement_id is not None else str(eval_subset),
                                        extention='csv')

            if os.path.exists(output_file):
                df = pd.read_csv(output_file)

                if eval_strategy == EvalStrategy.BY_AUDIO:

                    def most_common_value(series):
                        return collections.Counter(series).most_common(1)[0][0]

                    df = df.groupby('File').agg({
                        'Target': most_common_value,
                        'Prediction': most_common_value
                    }).reset_index()

                elif eval_strategy == EvalStrategy.BY_WINDOW:
                    pass

                else:
                    raise NotImplementedError(f'EvalStrategy has not is_trained implemented for \
                                            {eval_strategy}')

                df = df[['Target', 'Prediction']]
                return df

            os.makedirs(eval_base_dir, exist_ok=True)

            model = self.load(model_base_dir=model_base_dir)

            file_ids = []
            all_targets = []
            all_predictions = []

            for file_id in dataset.get_file_ids():
                samples, target = dataset.get_file_samples(file_id=file_id)

                predictions = self.predict(model=model, samples=samples)

                file_ids.extend([file_id] * len(samples))
                all_targets.extend([int(target)] * len(samples))
                all_predictions.extend(predictions.tolist())

            all_predictions = np.array(all_predictions)

            if len(all_predictions.shape) != 1:
                df = pd.DataFrame({"File": file_ids, "Target": all_targets})
                for idx in range(all_predictions.shape[1]):
                    df[f"Prediction_{idx}"] = all_predictions[:, idx]

                if 'Prediction' not in df.columns:
                    prediction_cols = [col for col in df.columns if col.startswith('Prediction_')]
                    df['Prediction'] = np.argmax(df[prediction_cols].values, axis=1)

            else:
                all_predictions = np.array([int(pred) for pred in all_predictions])
                df = pd.DataFrame({"File": file_ids,
                                   "Target": all_targets,
                                   "Prediction": all_predictions})

            df.to_csv(output_file, index=False)
            return self.eval(eval_subset = eval_subset,
                             eval_strategy = eval_strategy,
                             eval_base_dir = eval_base_dir,
                             model_base_dir = model_base_dir,
                             dataset = dataset,
                             complement_id = complement_id)

    def is_evaluated(self,
            eval_subset: Subset,
            eval_base_dir: str) -> bool:
        """
        Check if all models are evaluated for the specified dataset_id in the specified directory.

        Args:
            dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
            model_base_dir (str): The directory where the trained models are expected to be saved.

        Returns:
            bool: True if all models are evaluated in the directory,
                False otherwise.
        """
        output_file = self.output_filename(model_base_dir=eval_base_dir,
                                        complement=str(eval_subset),
                                        extention='csv')

        return os.path.exists(output_file)

class OptimizerTrainer(BaseTrainer):
    """Implementation of the BaseTrainer for training neural networks."""

    @staticmethod
    def default_optimizer_allocator(model: iara_model.BaseModel) -> torch.optim.Optimizer:
        """Allocate a default torch.optim.Optimizer for the given model, specifically the Adam
            optimizer, for the parameters of the provided model.

        Args:
            model (iara_model.BaseModel): The input model.

        Returns:
            torch.optim.Optimizer: The allocated optimizer.
        """
        return torch.optim.Adam(model.parameters())

    def __init__(self,
                 training_strategy: ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 model_allocator: typing.Callable[[typing.List[int], int],iara_model.BaseModel],
                 batch_size: int = 64 * 1024,
                 n_epochs: int = 512,
                 patience: int = 16,
                 optimizer_allocator: typing.Callable[[iara_model.BaseModel],
                                                      torch.optim.Optimizer]=None,
                 loss_allocator: typing.Callable[[torch.Tensor], torch.nn.Module]=None,
                 device: torch.device = iara.utils.get_available_device()) \
                     -> None:
        """Initialize the Trainer object with specified parameters.

        Args:
            training_strategy (TrainingStrategy): The training strategy to be used.
            n_targets (int): Number of targets in the training.
            model_allocator (typing.Callable[[typing.List[int], int], iara_model.BaseModel]):
                A callable that allocates the model with the given architecture.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            n_epochs (int, optional): The number of epochs for training. Defaults to 128.
            patience (int, optional): The patience for early stopping. None to execute all epochs.
                Default 10.
            optimizer_allocator (typing.Optional[typing.Callable[[iara_model.BaseModel],
                torch.optim.Optimizer]], optional): A callable that allocates the optimizer for
                the model. If provided, this callable will be used to allocate the optimizer.
                If not provided (defaulting to None), the Adam optimizer will be used.
            loss_allocator (typing.Optional[typing.Callable[[torch.Tensor], torch.nn.Module]],
                optional): A callable that allocates the loss function. If provided, this callable
                will be used to allocate the loss function. If not provided (defaulting to None),
                the default loss function corresponding to the specified training strategy will
                be used.
            device (torch.device, optional): The device for training
                (e.g., 'cuda' or 'cpu'). Defaults to iara.utils.get_available_device().
        """
        super().__init__(training_strategy, trainer_id, n_targets)
        self.model_allocator = model_allocator
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.optimizer_allocator = optimizer_allocator or \
                                    OptimizerTrainer.default_optimizer_allocator
        self.loss_allocator = loss_allocator or self.training_strategy.default_loss
        self.device = device

    def _prepare_for_training(self, trn_dataset: iara_dataset.BaseDataset) -> \
            typing.Dict[int,typing.Tuple[iara_model.BaseModel,
                                         torch.optim.Optimizer,
                                         torch.nn.Module]]:

        input_shape = list(trn_dataset[0][0].shape)
        trn_dict = {}

        if self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
            for target_id in range(self.n_targets):
                class_weights = self._class_weight(trn_dataset, target_id).to(self.device)

                model = self.model_allocator(input_shape, 1).to(self.device)
                optimizer = self.optimizer_allocator(model)
                loss = self.loss_allocator(class_weights)
                trn_dict[target_id] = model, optimizer, loss
            return trn_dict

        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            class_weights = self._class_weight(trn_dataset).to(self.device)

            model = self.model_allocator(input_shape, self.n_targets).to(self.device)
            optimizer = self.optimizer_allocator(model)
            loss = self.loss_allocator(class_weights)
            trn_dict[None] = model, optimizer, loss
            return trn_dict


        raise NotImplementedError('TrainingStrategy has not _loss implemented')

    def _check_dataset(self, dataset: iara_dataset.BaseDataset):
        unique_targets = set(torch.unique(dataset.get_targets()).int().tolist())
        expected_targets = set(torch.arange(self.n_targets).int().tolist())

        if not set(expected_targets).issubset(set(unique_targets)):
            print('unique_targets: ', unique_targets)
            print('expected_targets: ', expected_targets)
            raise UnboundLocalError(f'Targets in dataset not compatible with NNTrainer \
                                    configuration({self.n_targets})')

    def _export_trn(self, trn_error, batch_error, n_epochs, filename, log_scale = False):

        trn_batches = np.linspace(start=1, stop=n_epochs, num=len(trn_error))
        val_batches = np.linspace(start=1, stop=n_epochs, num=len(batch_error))

        plt.figure(figsize=(10, 5))
        plt.plot(trn_batches, trn_error, label='Training Loss')
        plt.plot(val_batches, batch_error, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.tight_layout()
        plt.legend()
        plt.grid()
        if log_scale:
            plt.semilogx()
        plt.savefig(filename)
        plt.close()

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        """Implementation of BaseTrainer.fit method, to fit (train) the model.

        Args:
            model_base_dir (str): The base directory to save any training-related outputs
                or artifacts.
            trn_dataset (iara_dataset.BaseDataset): The training dataset.
            val_dataset (iara_dataset.BaseDataset): The validation dataset.

        Returns:
            None
        """

        self._check_dataset(trn_dataset)
        self._check_dataset(val_dataset)
        if self.is_trained(model_base_dir=model_base_dir):
            return

        os.makedirs(model_base_dir, exist_ok=True)

        trn_loader = torch_data.DataLoader(trn_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True)
        val_loader = torch_data.DataLoader(val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True)

        container = self._prepare_for_training(trn_dataset=trn_dataset).items()

        partial_trn_model = self.output_filename(model_base_dir=model_base_dir,
                                             complement='partial',
                                             extention='pkl')
        if os.path.exists(partial_trn_model):
            with open(partial_trn_model, 'rb') as f:
                partial_trn = pickle.load(f)
        else:
            partial_trn = None


        for target_id, (model, optimizer, loss_module) in container if (len(container) == 1) else \
                    tqdm.tqdm(container, leave=False, desc="Classes", ncols=120):

            model_filename = self.output_filename(model_base_dir=model_base_dir,
                                              target_id=target_id)

            if os.path.exists(model_filename):
                continue

            best_val_loss = float('inf')
            epochs_without_improvement = 0
            best_model_state_dict = None

            trn_epoch_loss = []
            val_epoch_loss = []
            trn_batch_loss = []
            val_batch_loss = []
            n_epochs = 0

            if partial_trn is not None:
                if target_id is not None and partial_trn['target_id'] > target_id:
                    continue

                if partial_trn['target_id'] == target_id:
                    model = partial_trn['model']
                    optimizer = partial_trn['optimizer']
                    loss_module = partial_trn['loss_module']
                    start_epochs = partial_trn['epoch'] + 1
                    trn_epoch_loss = partial_trn['trn_epoch_loss']
                    val_epoch_loss = partial_trn['val_epoch_loss']
                    trn_batch_loss = partial_trn['trn_batch_loss']
                    val_batch_loss = partial_trn['val_batch_loss']
                    best_val_loss = partial_trn['best_val_loss']
                    epochs_without_improvement = partial_trn['epochs_without_improvement']
                    best_model_state_dict = partial_trn['best_model_state_dict']

            model = model.to(self.device)
            model.train()

            for i_epoch in tqdm.tqdm(range(self.n_epochs), leave=False, desc="Epochs", ncols=120):
                n_epochs += 1

                if partial_trn is not None:
                    if start_epochs >= i_epoch:
                        continue

                running_loss = []
                for samples, targets in tqdm.tqdm(trn_loader,
                                                  leave=False,
                                                  desc="Training Batchs", ncols=120):

                    optimizer.zero_grad()

                    if target_id is not None:
                        targets = torch.where(targets == target_id,
                                              torch.tensor(1.0),
                                              torch.tensor(0.0))

                    targets = targets.to(self.device)
                    samples = samples.to(self.device)

                    predictions = model(samples)

                    hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.n_targets).float()

                    loss = loss_module(predictions, hot_targets)
                    loss.backward()
                    trn_batch_loss.append(loss.item())
                    running_loss.append(loss.item())

                    optimizer.step()

                trn_epoch_loss.append(np.mean(running_loss))

                running_loss = []
                with torch.no_grad():
                    for samples, targets in tqdm.tqdm(val_loader,
                                                      leave=False,
                                                      desc="Evaluating Batch", ncols=120):

                        if target_id is not None:
                            targets = torch.where(targets == target_id,
                                                torch.tensor(1.0),
                                                torch.tensor(0.0))

                        targets = targets.to(self.device)
                        samples = samples.to(self.device)
                        predictions = model(samples)

                        hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.n_targets).float()

                        loss = loss_module(predictions, hot_targets)
                        running_loss.append(loss.item())
                        val_batch_loss.append(loss.item())

                    val_epoch_loss.append(np.mean(running_loss))

                running_loss = np.mean(running_loss)

                if running_loss < best_val_loss:
                    best_val_loss = running_loss
                    epochs_without_improvement = 0
                    best_model_state_dict = model.state_dict()
                else:
                    epochs_without_improvement += 1

                if self.patience is not None and epochs_without_improvement >= self.patience:
                    break

                state = {
                    'target_id': target_id,
                    'model': model,
                    'optimizer': optimizer,
                    'loss_module': loss_module,
                    'epoch': i_epoch,
                    'trn_epoch_loss': trn_epoch_loss,
                    'val_epoch_loss': val_epoch_loss,
                    'trn_batch_loss': trn_batch_loss,
                    'val_batch_loss': val_batch_loss,
                    'best_val_loss': best_val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                    'best_model_state_dict': best_model_state_dict,
                }
                with open(partial_trn_model, 'wb') as f:
                    pickle.dump(state, f)

                # trn_batch_loss_arr = np.array(trn_batch_loss)
                # val_batch_loss_arr = np.array(val_batch_loss)

                epoch_error_filename = self.output_filename(model_base_dir=model_base_dir,
                                                    target_id=target_id,
                                                    complement='trn_epochs',
                                                    extention='png')

                self._export_trn(trn_epoch_loss, val_epoch_loss,
                                 n_epochs,
                                 epoch_error_filename,
                                 log_scale=False)


            if best_model_state_dict:
                model.load_state_dict(best_model_state_dict)

            model.save(model_filename)

        if os.path.exists(partial_trn_model):
            os.remove(partial_trn_model)

    def predict(self,
                model: typing.Union[iara_model.BaseModel, typing.List[iara_model.BaseModel]],
                samples: torch.Tensor) -> torch.Tensor:

        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            model.eval()
            model = model.to(self.device)
            samples = samples.to(self.device)
            predictions = model(samples)
            predictions = predictions.cpu()

        # elif self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
            # predictions = torch.zeros((len(samples), len(model)))
            # for m_idx, m in enumerate(model):
            #     predictions[:, m_idx] = m(samples)

            # predictions = predictions.argmax(dim=1)

        else:
            raise NotImplementedError(f'TrainingStrategy has not predict implemented for \
                                    {self.training_strategy}')

        return predictions
    # def eval(self,
    #         dataset_id: str,
    #         eval_base_dir: str,
    #         model_base_dir: typing.Optional[str] = None,
    #         dataset: typing.Optional[iara_dataset.BaseDataset] = None) -> pd.DataFrame:
    #     """
    #     Implementation of BaseTrainer.eval method, to eval the model using the
    #         provided dataset.

    #     Args:
    #         dataset_id (str): Identifier for the dataset, e.g., 'val', 'trn', 'test'.
    #         eval_base_dir (str): The base directory to save any evaluation-related outputs
    #             or artifacts.
    #         model_base_dir typing.Optional(str): The base directory to save read trained models
    #             with non set the evaluation must already be done.
    #         dataset typing.Optional(iara_dataset.BaseDataset): The dataset to evaluate with non set
    #             the evaluation must already be done.

    #     Returns:
    #         pandas.DataFrame: DataFrame with two columns, ["Target", "Prediction"]
    #     """
    #     with torch.no_grad():

    #         all_predictions = []
    #         all_targets = []

    #         output_file = self.output_filename(model_base_dir=eval_base_dir,
    #                                        complement=dataset_id,
    #                                        extention='csv')

    #         if os.path.exists(output_file):
    #             df = pd.read_csv(output_file)

    #             prediction_columns = [col for col in df.columns if col.startswith('Prediction_')]
    #             max_prediction_index = np.argmax(df[prediction_columns].to_numpy(),axis=1)
    #             df_result = pd.DataFrame({'Prediction': max_prediction_index,
    #                                       'Target': df['Target']})

    #             return df_result

    #         os.makedirs(eval_base_dir, exist_ok=True)

    #         loader = torch_data.DataLoader(dataset, batch_size=self.batch_size)

    #         if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
    #             filename = self.output_filename(model_base_dir=model_base_dir)
    #             if not os.path.exists(filename):
    #                 raise FileNotFoundError(f"The model file '{filename}' does not exist. Ensure \
    #                                         that the model is trained before evaluating.")

    #             model = iara_model.BaseModel.load(filename)
    #             model.eval()

    #             for samples, targets in tqdm.tqdm(loader, leave=False, desc="Eval Batchs", ncols=120):
    #                 targets = targets.to(self.device)
    #                 samples = samples.to(self.device)
    #                 predictions = model(samples)

    #                 all_predictions.extend(predictions.cpu().tolist())
    #                 all_targets.extend(targets.cpu().tolist())


    #         elif self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
    #             models = []
    #             for target_id in range(self.n_targets):
    #                 filename = self.output_filename(model_base_dir=model_base_dir,
    #                                                 target_id=target_id)
    #                 if not os.path.exists(filename):
    #                     raise FileNotFoundError(f"The model file '{filename}' does not exist. \
    #                                             Ensure that the model is trained before \
    #                                             evaluating.")

    #                 model = iara_model.BaseModel.load(filename)
    #                 model.eval()
    #                 models.append(model)

    #             for samples, targets in tqdm.tqdm(loader, leave=False, desc="Eval Batchs", ncols=120):
    #                 samples = samples.to(self.device)

    #                 predictions = torch.zeros((len(targets), len(models)))
    #                 for model_idx, model in enumerate(models):
    #                     predictions[:, model_idx] = model(samples).cpu()

    #                 all_predictions.extend(predictions.tolist())
    #                 all_targets.extend(targets.cpu().tolist())

    #         else:
    #             raise NotImplementedError(f'TrainingStrategy has not is_trained implemented for \
    #                                     {self.training_strategy}')

    #         all_predictions = np.array(all_predictions)

    #         df = pd.DataFrame({"Target": all_targets})
    #         for idx in range(all_predictions.shape[1]):
    #             df[f"Prediction_{idx}"] = all_predictions[:, idx]

    #         prediction_columns = [col for col in df.columns if col.startswith('Prediction_')]
    #         max_prediction_index = np.argmax(df[prediction_columns].to_numpy(),axis=1)
    #         df_result = pd.DataFrame({'Prediction': max_prediction_index, 'Target': df['Target']})

    #         df.to_csv(output_file, index=False)
    #         return df_result

class RandomForestTrainer(BaseTrainer):
    """Implementation of the BaseTrainer for training a RandomForest networks."""

    def __init__(self,
                 training_strategy: ModelTrainingStrategy,
                 trainer_id: str,
                 n_targets: int,
                 n_estimators=100,
                 max_depth=None) \
                     -> None:
        super().__init__(training_strategy, trainer_id, n_targets)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self,
            model_base_dir: str,
            trn_dataset: iara_dataset.BaseDataset,
            val_dataset: iara_dataset.BaseDataset) -> None:
        """
        Method to fit (train) the model.

        This method implements the interface to fit (train) the model using the provided
            training and validation datasets.

        Args:
            model_base_dir (str): The base directory to save any training-related outputs
                or artifacts.
            trn_dataset (iara_dataset.BaseDataset): The training dataset.
            val_dataset (iara_dataset.BaseDataset): The validation dataset.
        """
        if self.is_trained(model_base_dir=model_base_dir):
            return

        os.makedirs(model_base_dir, exist_ok=True)

        if self.training_strategy == ModelTrainingStrategy.MULTICLASS:
            target_ids = [None]

        elif self.training_strategy == ModelTrainingStrategy.CLASS_SPECIALIST:
            target_ids = trn_dataset.get_targets()

        samples = trn_dataset.get_samples()

        if samples is None:
            raise UnboundLocalError("Training dataset without data")

        for target_id in target_ids:
            model_filename = self.output_filename(model_base_dir=model_base_dir,
                                                  target_id=target_id)

            if os.path.exists(model_filename):
                continue

            model = iara_forest.RandomForestModel(n_estimators=self.n_estimators,
                                                  max_depth=self.max_depth)

            targets = trn_dataset.get_targets()

            if target_id is not None:
                targets = torch.where(targets == target_id,
                                        torch.tensor(1.0),
                                        torch.tensor(0.0))

            model.fit(samples=samples, targets=targets)
            model.save(model_filename)
