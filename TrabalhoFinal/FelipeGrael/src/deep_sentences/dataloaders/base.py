from typing import Callable
from datasets import DatasetDict, load_dataset


class HFDataset:
    """
    Base class for Huggingface dataset wrappers with preprocessing capabilities.

    Provides a unified interface for loading, preprocessing, and tokenizing datasets
    from the Huggingface Hub.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str = "default",
        text_field: str = "text",
        cache_dir: str | None = None,
    ):
        """
        Initialize the dataset wrapper.

        Args:
            dataset_name: Name of the dataset on Huggingface Hub
            config_name: Configuration name for datasets with multiple configs
            split: Dataset split to load ('train', 'test', 'validation')
            cache_dir: Directory to cache the dataset
            **kwargs: Additional arguments passed to load_dataset
        """

        self._ds_orig: DatasetDict | None = None
        self._ds_preproc: DatasetDict | None = None
        self._ds_tokenized: DatasetDict | None = None

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_field = text_field
        self.cache_dir = cache_dir
        pass

    def preprocess(self, ds: DatasetDict) -> DatasetDict:
        """
        Apply dataset-specific preprocessing.

        Args:
            dataset: Raw dataset from Huggingface Hub

        Returns:
            Preprocessed dataset
        """
        return ds

    def load_and_preproces(self):
        if self._ds_orig is not None:
            return

        ds = load_dataset(self.dataset_name, self.dataset_config)
        assert isinstance(ds, DatasetDict)

        self._ds_orig = ds

        ds_preproc = self.preprocess(ds)
        self._ds_preproc = ds_preproc

    @property
    def dataset(self) -> DatasetDict:
        self.load_and_preproces()
        assert self._ds_preproc is not None
        return self._ds_preproc
