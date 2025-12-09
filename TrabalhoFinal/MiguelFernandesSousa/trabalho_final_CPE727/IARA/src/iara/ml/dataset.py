"""
Module providing standardized interfaces for training models in this library.
"""
import abc
import overrides
import typing
import tqdm
import math

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.utils.data as torch_data

import iara.utils
import iara.processing.manager as iara_proc_manager


class ExperimentDataLoader():
    """
    Custom dataset loader for audio data, should be use to pre-load all data in a experiment to RAM
        avoiding multiple load along side fold trainings:

    Attributes:
        MEMORY_LIMIT (int): Maximum size in bytes that can be loaded into memory.
            When a dataset exceeds this limit, the data is loaded partially as needed (Very low).
        N_WORKERS (int): Number of simultaneos threads to process run files and load data.
    """
    MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # bytes
    N_WORKERS = 8

    def __init__(self,
                 processor: iara_proc_manager.AudioFileProcessor,
                 file_ids: typing.Iterable[int],
                 targets: typing.Iterable,
                 central_offset_time: typing.Iterable[float] = None,
                 max_interval: int = 120 #seconds
                 ) -> None:
        """
        Args:
            processor (iara.processing.manager.AudioFileProcessor): An instance of the
                AudioFileProcessor class responsible for processing audio data.
            file_ids (typing.Iterable[int]): An iterable of integers representing the IDs of the
                audio files used in the search for file in the dataset.
            targets (typing.Iterable): An iterable representing the target labels corresponding to
                each audio file in the dataset.
        """
        self.processor = processor
        self.file_ids = file_ids
        self.targets = targets
        self.data_map = {}
        self.size_map = {}
        self.target_map = {}
        self.memory_map = {}
        self.total_memory = 0
        self.total_samples = 0
        self.central_offset_time = central_offset_time
        self.max_interval = max_interval

    def pre_load(self, file_ids = None) -> None:

        targets = [self.targets[index] for index, id in enumerate(self.file_ids) if id in file_ids]

        with ThreadPoolExecutor(max_workers=ExperimentDataLoader.N_WORKERS) as executor:
            futures = [executor.submit(self.__load, file_id, target) 
                       for file_id, target in zip(file_ids, targets)]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                    desc='Processing/Loading dataset', leave=False, ncols=120):
                future.result()

    def __get(self, file_id: int) -> pd.DataFrame:
        try:

            data_df, times = self.processor.get_data(file_id)
            times = np.array(times)

            if self.central_offset_time is not None:
                index = list(self.file_ids).index(file_id)
                offset =  list(self.central_offset_time)[index]

                try:
                    offset = int(offset) - self.max_interval/2
                except ValueError:
                    offset = times[0]

                indexes = np.where((times >= offset) & (times <= offset + self.max_interval))[0]

                return data_df.iloc[indexes]

            return data_df

        except Exception as e:
            # Tratar o erro capturando qualquer exceção
            print(f"Ocorreu um erro: {e}")

    def __load(self, file_id: int, target) -> None:
        """ Process or/and load a single file to data map

        Args:
            file_id (int): IDs of the audio files
            target (_type_): Target labels correspondent
        """        

        if file_id in self.size_map:
            return

        data_df = self.__get(file_id)

        memory = data_df.memory_usage(deep=True).sum()

        if self.total_memory < ExperimentDataLoader.MEMORY_LIMIT and \
                (self.total_memory + memory) > ExperimentDataLoader.MEMORY_LIMIT:
            self.data_map.clear()

        self.total_memory += memory

        if self.total_memory < ExperimentDataLoader.MEMORY_LIMIT:   
            self.data_map[file_id] = torch.tensor(data_df.values, dtype=torch.float32)

        self.total_samples += len(data_df)
        self.size_map[file_id] = len(data_df)
        self.target_map[file_id] = target
        self.memory_map[file_id] = memory

    def get(self, file_id: int, offset: int, n_samples: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Return data and target from offset sample of file_id

        Args:
            file_id (int): IDs of the audio files
            offset (int): Number lower than the number os windows generated in processing of
                file_id data

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: data, target
        """

        sample = self.get_all(file_id = file_id)[offset:offset+n_samples]
        target = torch.tensor(self.target_map[file_id], dtype=torch.int64)

        if n_samples != 1:
            sample = torch.unsqueeze(sample, dim=0)

        if n_samples == 1:
            sample = sample.squeeze(1)

        return sample, target

    def get_all(self, file_id: int) -> torch.Tensor:

        if file_id in self.data_map:
            return self.data_map[file_id]

        data_df = self.__get(file_id)
        data_df = torch.tensor(data_df.values, dtype=torch.float32)
        return data_df


    def __str__(self) -> str:
        return f'{self.total_samples} windows in {iara.utils.str_format_bytes(self.total_memory)}'

class BaseDataset(torch_data.Dataset):

    @abc.abstractmethod
    def get_targets(self) -> torch.tensor:
        """ Get all targets as tensors """

    @abc.abstractmethod
    def get_samples(self) -> torch.tensor:
        """ Get all sample as tensors """

    @abc.abstractmethod
    def get_file_samples(self, file_id: int) \
        -> typing.Tuple[torch.tensor, torch.tensor]:
        """ """

    @abc.abstractmethod
    def get_file_ids(self) -> typing.Iterable[int]:
        """ """

class InputType():
    def __init__(self,
                n_windows: int = 1,
                overlap: float = 0) -> None:
        self.n_windows = n_windows
        self.n_overlap = int(overlap * n_windows)
        self.n_news = n_windows - self.n_overlap

        if overlap < 0 or overlap > 1:
            raise UnboundLocalError(f'Invalid InputType, overlap({overlap}) should be between 0-1')

    def to_n_samples(self, qty_windows: int) -> int:
        return (qty_windows-self.n_windows)//self.n_news + 1

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InputType):
            return (self.n_windows == other.n_windows and
                    self.n_overlap == other.n_overlap and
                    self.n_news == other.n_news)
        return False

    def __str__(self) -> str:
        if self.n_windows == 1:
            return f'Window [{self.n_overlap}]'

        return f'Image [{self.n_windows},{self.n_overlap}]'
    
    def type_str(self) -> str:
        if self.n_windows == 1:
            return 'window'
        return 'image'


    @classmethod
    def Window(cls):
        return cls(n_windows=1, overlap=0)

    @classmethod
    def Image(cls, n_windows: int = 1, overlap: float = 0):
        return cls(n_windows=n_windows, overlap=overlap)

class AudioDataset(BaseDataset):

    def __init__(self,
                 loader: ExperimentDataLoader,
                 input_type: InputType,
                 file_ids: typing.Iterable[int]) -> None:
        self.loader = loader
        self.input_type = input_type
        # self.file_ids = file_ids
        self.file_ids = []
        self.limit_ids = [0]

        self.target_tensor = None
        self.sample_tensor = None

        loader.pre_load(file_ids)

        for file_id in file_ids:
            qty_windows = input_type.to_n_samples(loader.size_map[file_id])
            # self.limit_ids.append(self.limit_ids[-1] + qty_windows)

            if qty_windows > 0:
                self.limit_ids.append(self.limit_ids[-1] + qty_windows)
                self.file_ids.append(file_id)

    def __len__(self):
        return self.limit_ids[-1]

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        current_id = next(i for i, valor in enumerate(self.limit_ids) if valor > index) - 1
        offset_index = index - self.limit_ids[current_id]

        return self.loader.get(self.file_ids[current_id],
                               offset_index * self.input_type.n_news,
                               self.input_type.n_windows)

    def __str__(self) -> str:
        total_memory = 0
        for file_id in self.file_ids:
            total_memory += self.loader.memory_map[file_id]
        return f'{self.limit_ids[-1]} windows in {iara.utils.str_format_bytes(total_memory)}'

    @overrides.override
    def get_targets(self) -> torch.tensor:

        if self.target_tensor is None:

            self.target_tensor = torch.tensor([], dtype=torch.float32)

            for file_id in self.file_ids:
                target = self.loader.target_map[file_id]
                size = self.loader.size_map[file_id]

                self.target_tensor = torch.cat([self.target_tensor,
                                        torch.tensor([target] * size, dtype=torch.int32)])

        return self.target_tensor

    @overrides.override
    def get_samples(self) -> torch.tensor:

        if self.sample_tensor is None:

            self.sample_tensor = torch.tensor([], dtype=torch.float32)

            for file_id in self.file_ids:
                self.sample_tensor = torch.cat([self.sample_tensor, self.loader.get_all(file_id)])

        return self.sample_tensor

    @overrides.override
    def get_file_samples(self, file_id: int) \
        -> typing.Tuple[torch.tensor, torch.tensor]:

        samples = []

        base_file_index = self.file_ids.index(file_id)

        for index in range(self.limit_ids[base_file_index+1] - self.limit_ids[base_file_index]):
            sample, target = self[index + self.limit_ids[base_file_index]]
            samples.append(sample)

        return torch.stack(samples), target

    @overrides.override
    def get_file_ids(self) -> typing.Iterable[int]:
        return self.file_ids