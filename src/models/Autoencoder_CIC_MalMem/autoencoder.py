import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Autoencoder (AE) usado para pré-treinamento não supervisionado (SAE).
    Recebe as 16 features filtradas e tenta reconstruí-las.
    """
    def __init__(self, input_shape, num_classes, config):
        super(Autoencoder, self).__init__()
        
        # input_shape deve ser um inteiro (16 features)
        if isinstance(input_shape, tuple):
             self.input_size = input_shape[0] * input_shape[1] 
        else:
             self.input_size = input_shape # Deve ser 16
        
        self.enc1 = nn.Linear(self.input_size, 256) # <--- CORREÇÃO: 256 Neurônios
        self.bn1 = nn.BatchNorm1d(256)
        self.enc2 = nn.Linear(256, 128)           # <--- CORREÇÃO: 128 Neurônios
        self.bn2 = nn.BatchNorm1d(128)
        self.enc_code = nn.Linear(128, 64)         # Camada Latente (Code) 64 Neurônios
        
        # --- DECODER (64 -> 16) ---
        # Note que o Decoder é simétrico ao Encoder
        self.dec3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dec4 = nn.Linear(128, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dec_out = nn.Linear(256, self.input_size) # Saída deve ser do mesmo tamanho da entrada (16)

    def forward(self, x):
        # x shape: (Batch, 16)
        
        # Encoder
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        
        # Camada Latente
        code = self.enc_code(x) 
        
        # Decoder
        x = F.relu(self.bn3(self.dec3(code)))
        x = F.relu(self.bn4(self.dec4(x)))
        
        # Saída (Sem ReLU na saída para permitir valores negativos/positivos, crucial para dados escalonados)
        reconstruction = self.dec_out(x)
        
        return reconstruction

    def get_code(self, x):
        """Função para obter a representação latente (útil para feature extraction)."""
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        return self.enc_code(x)