"""
Dataset loader para Shenzhen Hospital X-ray Set
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ShenzhenTBDataset(Dataset):
    """
    Dataset customizado para carregar imagens de raio-X do dataset Shenzhen
    para detecção de tuberculose.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[A.Compose] = None,
        mode: str = 'train'
    ):
        """
        Args:
            data_dir: Diretório contendo as imagens
            image_size: Tamanho para redimensionar as imagens (altura, largura)
            transform: Transformações a serem aplicadas
            mode: 'train', 'val', ou 'test'
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode
        
        # Carregar lista de imagens
        self.image_paths = []
        self.labels = []
        
        # Diretórios esperados: normal/ e tuberculosis/
        normal_dir = os.path.join(data_dir, 'normal')
        tb_dir = os.path.join(data_dir, 'tuberculosis')
        
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)  # Normal = 0
        
        if os.path.exists(tb_dir):
            for img_name in os.listdir(tb_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(tb_dir, img_name))
                    self.labels.append(1)  # Tuberculosis = 1
        
        # Definir transformações padrão se não fornecidas
        if transform is None:
            self.transform = self._get_default_transforms(mode)
        else:
            self.transform = transform
    
    def _get_default_transforms(self, mode: str) -> A.Compose:
        """Define transformações padrão baseadas no modo"""
        if mode == 'train':
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retorna uma imagem e seu rótulo"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Carregar imagem
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Aplicar transformações
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Retorna a distribuição das classes"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(['Normal', 'Tuberculosis'], counts))


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria dataloaders para treino, validação e teste
    
    Args:
        data_dir: Diretório raiz dos dados
        batch_size: Tamanho do batch
        image_size: Tamanho das imagens
        train_split: Proporção de dados para treino
        val_split: Proporção de dados para validação
        num_workers: Número de workers para carregamento
        seed: Seed para reprodutibilidade
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split
    
    # Criar dataset completo
    full_dataset = ShenzhenTBDataset(
        data_dir=data_dir,
        image_size=image_size,
        mode='train'
    )
    
    # Calcular tamanhos dos splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Dividir dataset
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
