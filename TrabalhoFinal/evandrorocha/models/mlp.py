"""
Multi-Layer Perceptron (MLP) para Detecção de Tuberculose
Implementação com extração de features usando Transfer Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FeatureExtractorCNN(nn.Module):
    """
    Extrator de features usando CNN pré-treinada (ResNet50)
    Extrai features de alta dimensão das imagens de raio-X
    """
    def __init__(self, pretrained=True, freeze_layers=True):
        super(FeatureExtractorCNN, self).__init__()
        
        # Carrega ResNet50 pré-treinada no ImageNet
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove a última camada FC (classificador)
        # Mantém apenas as camadas convolucionais
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Congela os pesos se necessário (para usar apenas como extrator)
        if freeze_layers:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Tensor de imagens [batch_size, 3, 224, 224]
        Returns:
            features: Tensor de features [batch_size, 2048]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # [batch_size, 2048]
        return x


class MLP_TB_Detector(nn.Module):
    """
    Multi-Layer Perceptron para classificação binária (TB vs Normal)
    
    Arquitetura:
    - Input: 2048 features (extraídas do ResNet50)
    - Hidden Layer 1: 512 neurônios + BatchNorm + ReLU + Dropout
    - Hidden Layer 2: 256 neurônios + BatchNorm + ReLU + Dropout
    - Hidden Layer 3: 128 neurônios + BatchNorm + ReLU + Dropout
    - Output: 2 classes (Normal, TB)
    """
    def __init__(self, input_size=2048, hidden_sizes=[512, 256, 128], 
                 num_classes=2, dropout_rate=0.5):
        super(MLP_TB_Detector, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Camadas do MLP
        layers = []
        
        # Primeira camada oculta
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        
        # Camadas ocultas intermediárias
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
        
        # Camada de saída
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de features [batch_size, input_size]
        Returns:
            output: Logits [batch_size, num_classes]
        """
        return self.mlp(x)


class MLP_TB_Complete(nn.Module):
    """
    Modelo completo: Feature Extractor + MLP
    Pode ser usado de duas formas:
    1. End-to-end: treina tudo junto
    2. Two-stage: extrai features primeiro, depois treina MLP
    """
    def __init__(self, use_pretrained=True, freeze_extractor=True,
                 hidden_sizes=[512, 256, 128], dropout_rate=0.5):
        super(MLP_TB_Complete, self).__init__()
        
        # Extrator de features
        self.feature_extractor = FeatureExtractorCNN(
            pretrained=use_pretrained,
            freeze_layers=freeze_extractor
        )
        
        # MLP classificador
        self.mlp = MLP_TB_Detector(
            input_size=2048,
            hidden_sizes=hidden_sizes,
            num_classes=2,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de imagens [batch_size, 3, 224, 224]
        Returns:
            output: Logits [batch_size, 2]
        """
        features = self.feature_extractor(x)
        output = self.mlp(features)
        return output
    
    def extract_features(self, x):
        """
        Extrai apenas as features sem classificar
        Útil para análise e visualização
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features


class SimpleMLP(nn.Module):
    """
    MLP Simples para comparação
    Usa menos camadas e neurônios
    """
    def __init__(self, input_size=2048, hidden_size=256, num_classes=2, dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


def get_model(model_type='complete', **kwargs):
    """
    Factory function para criar modelos
    
    Args:
        model_type: 'complete', 'mlp_only', 'simple'
        **kwargs: argumentos específicos do modelo
    
    Returns:
        model: Modelo PyTorch
    """
    if model_type == 'complete':
        return MLP_TB_Complete(**kwargs)
    elif model_type == 'mlp_only':
        return MLP_TB_Detector(**kwargs)
    elif model_type == 'simple':
        return SimpleMLP(**kwargs)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")


if __name__ == "__main__":
    # Teste dos modelos
    print("=" * 60)
    print("Testando Modelos MLP para Detecção de TB")
    print("=" * 60)
    
    # Teste do modelo completo
    print("\n1. Modelo Completo (Feature Extractor + MLP):")
    model_complete = MLP_TB_Complete()
    dummy_images = torch.randn(4, 3, 224, 224)  # Batch de 4 imagens
    output = model_complete(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Número de parâmetros: {sum(p.numel() for p in model_complete.parameters()):,}")
    print(f"   Parâmetros treináveis: {sum(p.numel() for p in model_complete.parameters() if p.requires_grad):,}")
    
    # Teste do MLP apenas
    print("\n2. MLP Apenas (requer features pré-extraídas):")
    model_mlp = MLP_TB_Detector()
    dummy_features = torch.randn(4, 2048)  # Batch de 4 vetores de features
    output = model_mlp(dummy_features)
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Número de parâmetros: {sum(p.numel() for p in model_mlp.parameters()):,}")
    
    # Teste do MLP simples
    print("\n3. MLP Simples:")
    model_simple = SimpleMLP()
    output = model_simple(dummy_features)
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Número de parâmetros: {sum(p.numel() for p in model_simple.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Testes concluídos com sucesso!")
    print("=" * 60)
