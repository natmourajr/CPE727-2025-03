import os
from typing import Callable, Optional, Tuple, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# BASE_DIR = pasta FelipeAndrade
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR = os.path.join(BASE_DIR, "data", "crops")


class Mbgv2CropsDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ):
        """
        split: 'train', 'val' ou 'test'
        transform: transforms do torchvision (Resize, ToTensor, aug, etc.)
        """
        assert split in {"train", "val", "test"}, f"split inválido: {split}"

        self.split = split
        self.transform = transform

        csv_path = os.path.join(CROPS_DIR, f"{split}.csv")
        if not os.path.isfile(csv_path):
            raise RuntimeError(f"CSV de split não encontrado: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Mapeia labels string -> ids numéricos
        self.classes: List[str] = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Converte coluna de labels para ids
        self.labels = self.df["label"].map(self.class_to_idx).values
        self.filepaths = self.df["filepath"].values

        print(f"[Mbgv2CropsDataset] split={split}, amostras={len(self.df)}")
        print(f"[Mbgv2CropsDataset] classes={self.classes}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path = self.filepaths[idx]
        label_id = int(self.labels[idx])

        # Caminho absoluto da imagem
        img_path = os.path.join(BASE_DIR, rel_path)

        # Carrega imagem RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback simples: converte para tensor [0,1]
            img = torch.from_numpy(
                np.array(img).transpose(2, 0, 1)
            ).float() / 255.0  # type: ignore[name-defined]

        return img, label_id
