from ThreeWToolkit.dataset import ParquetDataset, ParquetDatasetConfig

config = ParquetDatasetConfig(path="./dataset")  # folder where it will be stored
dataset = ParquetDataset(config)
