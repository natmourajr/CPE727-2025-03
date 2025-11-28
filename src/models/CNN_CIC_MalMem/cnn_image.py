import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Image(nn.Module):
    """
    CNN adaptada para classificação de features de memória 8x8 (1 Canal).
    A arquitetura foi simplificada para evitar o colapso do mapa de features.
    """
    # Alteração 1: Mudança da variável 'model_config' para 'config' e 'input_shape'
    def __init__(self, input_shape, num_classes, config):
        super(CNN_Image, self).__init__()
        
        # input_shape é uma tupla (H, W), e input_channels é fixo em 1 para escala de cinza.
        self.height, self.width = input_shape 
        self.num_classes = num_classes
        
        model_config = config['architectures']['DeepNN_MLP']

        dropout_p = model_config.get('dropout_rate', 0.5)
        
        # --- Camadas Convolucionais ---
        
        # Alteração 2: In_channels = 1 (Escala de Cinza)
        # Bloco 1: Input (B, 1, 8, 8)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        
        # Alteração 3: Removido MaxPool2d do Bloco 1
        # Usar MaxPool em 8x8 faria a dimensão cair para 4x4, inviabilizando mais camadas.
        
        # Bloco 2: (16, 8, 8) -> (32, 8, 8)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Alteração 4: Adicionado MaxPool2d AQUI (dimensão cai para 4x4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Removido Bloco 3 e Bloco 4 (não há espaço para mais camadas de Pooling/Convolução)

        # Alteração 5: Cálculo Manual da Saída Linear
        # Saída após Bloco 2 e Pool: 32 canais * 4 altura * 4 largura = 512
        self.linear_input_size = 32 * 4 * 4 # 512
            
        print(f"Tamanho de entrada da camada linear: {self.linear_input_size}") 
        
        # --- Camadas Densas ---
        self.fc1 = nn.Linear(self.linear_input_size, 128) # Reduzido para 128 (era 512)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        """Helper para passar os dados pelas camadas convolucionais."""
        # x shape: (B, 1, 8, 8)
        
        # Bloco 1: Conv -> BN -> ReLU (Sem Pool)
        x = F.relu(self.bn1(self.conv1(x))) # x shape: (B, 16, 8, 8)
        
        # Bloco 2: Conv -> BN -> ReLU
        x = F.relu(self.bn2(self.conv2(x))) # x shape: (B, 32, 8, 8)
        
        # Pool: Achatamento final (dimensão cai para 4x4)
        x = self.pool(x) # x shape: (B, 32, 4, 4)
        
        return x

    def forward(self, x):
        # Passa pelos blocos convolucionais
        x = self._forward_features(x)
        
        # Flatten: Prepara para as camadas densas
        x = x.view(x.size(0), -1) # x shape: (Batch, 512)
        
        # Camadas Densas
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        
        return x