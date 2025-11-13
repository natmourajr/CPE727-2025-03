import torch.nn as nn
import torch.nn.functional as F

class DeepNN_MLP(nn.Module):
    """
    Deep Neural Network (Multi-Layer Perceptron) para classificação de dados tabulares (Vetoriais).
    Aceita input_shape universal.
    """
    # Renomeamos input_size para input_shape para compatibilidade com o evaluator.py
    def __init__(self, input_shape, num_classes, config): 
        super(DeepNN_MLP, self).__init__()
        
        # O input_shape para o MLP é um inteiro (o número de features: 55)
        if isinstance(input_shape, tuple):
             # Isso só aconteceria se o MLP fosse chamado com dados de imagem (o que não queremos)
             self.input_size = input_shape[0] * input_shape[1] 
        else:
             self.input_size = input_shape # Deve ser 55
             
        self.num_classes = num_classes
        dropout_p = config['model'].get('dropout_rate', 0.5)
        
        # --- Camadas Densas e Regularização ---
        
        # Camada 1: Entrada -> 256
        self.fc1 = nn.Linear(self.input_size, 256) # Usa self.input_size
        self.bn1 = nn.BatchNorm1d(256) 
        self.drop1 = nn.Dropout(p=dropout_p)
        
        # Camada 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=dropout_p)

        # Camada 3: 128 -> 64
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Camada de Saída
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (Batch, 55)
        
        # Camada 1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        # Camada 2
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        # Camada 3
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.fc_out(x)
        return x