import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class Normalization:
    """
    Normalization
    =============

    Classe to implement normalization transformation

    Input:
        - mean: Mean value of every feature
        - std: Standard deviation value of every feature
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean.values, dtype=torch.float32)
        self.std  = torch.tensor(std.values,  dtype=torch.float32)


    def __call__(self, x):
        return (x - self.mean) / self.std


class DsaDataset(Dataset):
    """
    DsaDataset
    ==========

    Class to implement a custom dataset built for DSA data
    """
    def __init__(self, df, feature_col, target_col, transform = None):
        self.X = torch.tensor(df[feature_col].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

        self.transform = transform

    
    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)
        
        return X, y
    

def clean_data(data, X_col):
    """
    Clean Data
    ==========

    Function to clean dataset from nan values and columns with null standard deviation

    Input:
        - X_col: list of dataset's feature columns
    """

    cols_with_nan = data[X_col].columns[data[X_col].isna().any(axis=0)]
    if len(cols_with_nan) > 0:
        X_col = [c for c in X_col if c not in cols_with_nan]

    stds = data[X_col].std(axis=0)
    zero_std_cols = stds[stds == 0].index.tolist()
    if len(zero_std_cols) > 0:
        X_col = [c for c in X_col if c not in zero_std_cols]

    return X_col


def build_dataloaders(data_path, feature_exclude, target_col, batch_size, train_size, val_size, random_state):
    """
    Build Dataloaders
    =================

    Main function to build Torch DataLoaders

    Input:
        - data_path: Directory to the original CSV data
        - feature_exclude: Columns to exclude in order to build feature's DataFrame
        - target_col: Target column
        - batch_size: Size of the batches
        - train_size: Size of the training dataset
        - val_size: Size of the validation dataset. The size of the test data corresponds to val_size/(1 - train_size)
        - random_state: Random state to the train-test split
    """

    data = pd.read_csv(data_path, index_col=0)

    X_col = [c for c in data.columns if c not in feature_exclude]
    y_col = target_col

    data[X_col] = data[X_col].apply(pd.to_numeric, errors='coerce')

    X_col = clean_data(data, X_col)

    df_train, df_temp = train_test_split(data, test_size=round(1 - train_size, 1), random_state=random_state)
    df_val, df_test = train_test_split(df_temp, test_size=round(val_size/(1 - train_size), 1), random_state=random_state)

    mean = df_train[X_col].mean(axis = 0)
    std  = df_train[X_col].std(axis = 0) + 1e-8

    transform = Normalization(mean, std)

    train_ds = DsaDataset(df_train, X_col, y_col, transform)
    val_ds   = DsaDataset(df_val, X_col, y_col, transform)
    test_ds  = DsaDataset(df_test, X_col, y_col, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, X_col