import os
import json
import torch
import pandas as pd

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np



class ImageNormalization:
    """
    Image Normalization
    ===================

    Classe to implement normalization transformation for images

    Input:
        - mean_diag: Mean value of diagonal elements
        - std_diag: Standard deviation value of diagonal elements
        - mean_off: Mean value of off-diagonal elements
        - std_off: Standard deviation value of off-diagonal elements
    """

    def __init__(self, mean_diag, std_diag, mean_off, std_off, eps=1e-8):

        self.mean_diag = mean_diag.view(-1, 1, 1)
        self.std_diag  = std_diag.view(-1, 1, 1) + eps

        self.mean_off  = mean_off.view(-1, 1, 1)
        self.std_off   = std_off.view(-1, 1, 1) + eps


    def __call__(self, img):

        diag_mask = torch.zeros_like(img, dtype=torch.bool)
        
        idx = torch.arange(img.size(1))
        diag_mask[:, idx, idx] = True # Get diagonal elements

        off_mask = ~diag_mask # Get off-diagonal elements

        out = torch.empty_like(img)

        diag_norm = (img - self.mean_diag) / self.std_diag # Normalize diagonal elements
        out[diag_mask] = diag_norm[diag_mask]

        off_norm = (img - self.mean_off) / self.std_off # Normalize off-diagonal elements
        out[off_mask] = off_norm[off_mask]

        return out
    

def get_statistics(imgs):
    """
    Function to get image's statistics for diagonal and off-diagonal elements
    """

    N, C, H, W = imgs.shape

    idx = torch.arange(H)

    diag_vals = imgs[:, :, idx, idx]

    mean_diag = diag_vals.mean(dim=(0, 2))
    std_diag  = diag_vals.std(dim=(0, 2))

    eye = torch.eye(H, dtype=torch.bool, device=imgs.device)
    off_mask_2d = ~eye

    off_vals = imgs[:, :, off_mask_2d].view(N, C, -1)

    mean_off = off_vals.mean(dim=(0, 2))
    std_off  = off_vals.std(dim=(0, 2))

    eps = 1e-8
    std_diag += eps
    std_off  += eps

    return mean_diag, std_diag, mean_off, std_off


class DsaImageDataset(Dataset):
    """
    DsaImageDataset
    ===============
    Dataset para dados de DSA em formato de imagem (C, H, W).
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        else:
            img = img.float()

        if self.transform:
            img = self.transform(img)

        y = self.labels[idx]
        return img, y


def build_images(data, ybus_tuples, bus_names, buses):
    """
    Build Images
    ============

    Build images from base case power flow data.

    The channels are:
        - (1) Bus voltages
        - (2) Bus angles
        - (3) MVA power flow

    Input:
        - data: Power flow dataset
        - ybus_tuples: list of YBus tuples to extract connectivity between buses
        - bus_names: dict of bus name per bus number
        - buses: list of buses available in the dataset
    Output:
        - img_dataset: Power flow image dataset
    """
    n = len(buses)

    it = 0
    img_dataset = []
    for _, data_row in data.iterrows():

        v_mat   = np.zeros((n, n), dtype=np.float32)
        d_mat   = np.zeros((n, n), dtype=np.float32)
        mva_mat = np.zeros((n, n), dtype=np.float32)

        for _, row in ybus_tuples.iterrows():
            try:
                i_bus = row['Bus(Row)']
                j_bus = row['Bus(Col)']

                i = buses.index(i_bus)
                j = buses.index(j_bus)

                i_name = bus_names[str(int(i_bus))]
                j_name = bus_names[str(int(j_bus))]

                v_i_col   = f"{int(i_bus)}_{i_name}_V"
                v_j_col   = f"{int(j_bus)}_{j_name}_V"
                ang_i_col = f"{int(i_bus)}_{i_name}_d"
                ang_j_col = f"{int(j_bus)}_{j_name}_d"

                if i_bus == j_bus:
                    v_mat[i, j] = data_row[v_i_col]
                    d_mat[i, j] = data_row[ang_i_col]
                else:
                    v_mat[i, j] = data_row[v_i_col]   - data_row[v_j_col]
                    d_mat[i, j] = data_row[ang_i_col] - data_row[ang_j_col]

            except (ValueError, KeyError):
                continue

        for _, row in ybus_tuples[ybus_tuples['Bus(Row)'] != ybus_tuples['Bus(Col)']].iterrows():
            try:
                i_bus = row['Bus(Row)']
                j_bus = row['Bus(Col)']

                i = buses.index(i_bus)
                j = buses.index(j_bus)

                i_name = bus_names[str(int(i_bus))]
                j_name = bus_names[str(int(j_bus))]

                key_ij = f"{i_name}_{j_name}_MVA"
                try:
                    val = data_row[key_ij]
                    mva_mat[i, j] += val
                    mva_mat[i, i] += val
                except KeyError:
                    key_ji = f"{j_name}_{i_name}_MVA"
                    val = data_row[key_ji]
                    mva_mat[j, i] += val
                    mva_mat[j, j] += val

            except (ValueError, KeyError):
                continue

        pf_img = np.stack([v_mat, d_mat, mva_mat], axis=0)
        pf_img = torch.from_numpy(pf_img)

        img_dataset.append(pf_img)
        it += 1
        print(f'Loading image dataset - {round(it/data.shape[0]*100)}%')
    
    return img_dataset


def build_image_dataloaders(data_path, feature_exclude, target_col, batch_size, train_size, val_size, random_state):
    """
    Build Image Dataloaders
    =======================

    Main function to build Torch DataLoaders from image power flow data

    Input:
        - data_path: Directory to the original CSV data
        - feature_exclude: Columns to exclude in order to build feature's DataFrame (se ainda precisar)
        - target_col: Target column
        - batch_size: Size of the batches
        - train_size: Size of the training dataset
        - val_size: Size of the validation dataset. The size of the test data corresponds to val_size/(1 - train_size)
        - random_state: Random state to the train-test split
    """

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'buses_MT.json'), 'r') as file:
        bus_names = json.load(file)

    buses = [int(bus) for bus in list(bus_names.keys())]

    ybus = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'DumpYBus.dat'), sep=';')
    ybus_tuples = ybus[ybus['Bus(Row)'].isin(buses)][['Bus(Row)', 'Bus(Col)']]

    data = pd.read_csv(data_path, index_col=0) 

    X_col = [c for c in data.columns if c not in feature_exclude]
    y_col = target_col

    data[X_col] = data[X_col].apply(pd.to_numeric, errors='coerce')

    df_train, df_temp = train_test_split(data, test_size=round(1 - train_size, 1), random_state=random_state)
    df_val, df_test = train_test_split(df_temp, test_size=round(val_size / (1 - train_size), 1), random_state=random_state)

    # Build Train, Validation and Test datasets:
    train_img = build_images(df_train, ybus_tuples, bus_names, buses)
    val_img   = build_images(df_val, ybus_tuples, bus_names, buses)
    test_img  = build_images(df_test, ybus_tuples, bus_names, buses)

    train_img_tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in train_img], dim=0)

    mean_diag, std_diag, mean_off, std_off = get_statistics(train_img_tensor)

    img_transform = ImageNormalization(
        mean_diag=mean_diag,
        std_diag=std_diag,
        mean_off=mean_off,
        std_off=std_off
    )

    y_train = df_train[y_col].values
    y_val   = df_val[y_col].values
    y_test  = df_test[y_col].values

    train_ds = DsaImageDataset(train_img, y_train, transform=img_transform)
    val_ds   = DsaImageDataset(val_img,   y_val,   transform=img_transform)
    test_ds  = DsaImageDataset(test_img,  y_test,  transform=img_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader