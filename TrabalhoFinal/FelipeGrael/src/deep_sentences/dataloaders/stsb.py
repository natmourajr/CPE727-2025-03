"""
Loader for MTEB/STS-b dataset
"""
from datasets import DatasetDict, concatenate_datasets

from .base import HFDataset


class STSBDataset(HFDataset):

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        super().__init__(
            dataset_name='mteb/stsbenchmark-sts',
            dataset_config='default',
            text_field='text',
            cache_dir=cache_dir,
        )

    def preprocess(self, ds: DatasetDict) -> DatasetDict:

        dev_set = concatenate_datasets([ds['train'], ds['validation']])

        new_ds = DatasetDict({
            'dev': dev_set,
            'test': ds['test'],
        })

        return new_ds
