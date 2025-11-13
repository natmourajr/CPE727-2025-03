import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Image(nn.Module):
    """
    Convolutional Neural Network (CNN) para classificação de dados 2D (224x224).
    Esta arquitetura é mais profunda para lidar com o tamanho da imagem de entrada.
    """
    def __init__(self, model_config, num_classes, device="cpu"):
        super(CNN_Image, self).__init__()
        
        self.num_classes = num_classes
        dropout_p = model_config['model'].get('dropout_rate', 0.25)
        
        # --- Camadas Convolucionais ---
        
        # Bloco 1: Input (B, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # CORREÇÃO 1: in_channels=3
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Saída: (B, 16, 112, 112)
        
        # Bloco 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Saída: (B, 32, 56, 56)
        
        # Bloco 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Saída: (B, 64, 28, 28)

        # Bloco 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # Saída: (B, 128, 14, 14)
        
        # --- Cálculo automático da camada densa ---
        # CORREÇÃO 2: Calcular o tamanho de entrada do FC dinamicamente
        
        # Criamos uma entrada "dummy" para descobrir o tamanho de saída
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) # 1 amostra, 3 canais, 224x224
            dummy_output = self._forward_features(dummy_input)
            self.linear_input_size = dummy_output.view(1, -1).size(1) # Pega o tamanho achatado
            
        print(f"Tamanho de entrada da camada linear calculado: {self.linear_input_size}") # Ex: 128 * 14 * 14
        
        # --- Camadas Densas ---
        self.fc1 = nn.Linear(self.linear_input_size, 512) # Usa o tamanho calculado
        self.drop = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(512, num_classes)

    def _forward_features(self, x):
        """Helper para passar os dados pelas camadas convolucionais."""
        # Bloco 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Bloco 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Bloco 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Bloco 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        
        # Passa pelos blocos convolucionais
        x = self._forward_features(x)
        
        # Flatten: Prepara para as camadas densas
        # O tamanho será (Batch, 128 * 14 * 14) = (Batch, 25088)
        x = x.view(x.size(0), -1) 
        
        # Camadas Densas
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        
        return x