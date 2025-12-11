"""
Configurações e hiperparâmetros do projeto
"""

# Configurações de Dataset
DATA_CONFIG = {
    'data_dir': './data/shenzhen',
    'image_size': (224, 224),
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'num_workers': 4,
    'seed': 42
}

# Configurações de Treinamento
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'device': 'cuda'  # ou 'cpu'
}

# Configurações de Modelos
MODEL_CONFIGS = {
    'resnet50': {
        'model_name': 'resnet50',
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False
    },
    'resnet101': {
        'model_name': 'resnet101',
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False
    },
    'densenet121': {
        'model_name': 'densenet121',
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False
    },
    'densenet169': {
        'model_name': 'densenet169',
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False
    },
    'efficientnet_b0': {
        'model_name': 'efficientnet_b0',
        'pretrained': True,
        'dropout': 0.4,
        'freeze_backbone': False
    },
    'vgg16': {
        'model_name': 'vgg16',
        'pretrained': True,
        'dropout': 0.5,
        'freeze_backbone': False
    }
}

# Configurações de Data Augmentation
AUGMENTATION_CONFIG = {
    'train': {
        'horizontal_flip': 0.5,
        'rotation_limit': 15,
        'shift_limit': 0.0625,
        'scale_limit': 0.1,
        'brightness_contrast': 0.3,
    },
    'val_test': {
        # Apenas normalização para validação e teste
    }
}

# Configurações de Otimização
OPTIMIZER_CONFIG = {
    'type': 'adam',
    'lr': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-5
}

# Configurações de Learning Rate Scheduler
SCHEDULER_CONFIG = {
    'type': 'reduce_on_plateau',
    'mode': 'max',
    'factor': 0.5,
    'patience': 5,
    'verbose': True
}

# Configurações de Salvamento
SAVE_CONFIG = {
    'save_dir': './models',
    'log_dir': './runs',
    'results_dir': './results',
    'save_best_only': True,
    'save_frequency': 5  # Salvar checkpoint a cada N épocas
}

# Métricas para monitoramento
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc_roc',
    'confusion_matrix'
]
