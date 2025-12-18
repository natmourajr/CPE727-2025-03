import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DeepNN_DBN_Classifier(nn.Module):
    """
    MLP que carrega os pesos pré-treinados da Deep Belief Network (DBN)
    e é refinada (fine-tuned) de forma supervisionada.
    """
    def __init__(self, input_shape, num_classes, config, device='cuda', pretrain_weights_path=None):
        super(DeepNN_DBN_Classifier, self).__init__()
        self.device = device
        self.pretrain_weights_path = pretrain_weights_path
        # Arquitetura MLP (Encoder DBN): 16 -> 256 -> 128 -> 64
        self.layer1 = nn.Linear(input_shape, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64) # Camada Latente
        
        # Camada de Classificação
        self.classifier_head = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config['training']['dropout_rate'])
        
        self.device = device
        
    def forward(self, x):
        x = x.float() # Garante float32
        
        # Camadas Ocultas
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        
        # Camada de Saída
        x = self.classifier_head(x)
        return x

    def load_pretrained_weights(self, dbn_weights_path: Path):
        """
        Carrega os pesos pré-treinados de cada RBM (W e h_bias) para as camadas MLP.
        
        dbn_weights_path: Caminho para o arquivo .pt que contém o state_dict da DBN.
        """
        print(f"\n--- CARREGANDO PESOS PRÉ-TREINADOS DA DBN de: {dbn_weights_path.name} ---")
        
        dbn_state_dict = torch.load(dbn_weights_path, map_location=self.device)
        
        # Mapeamento dos pesos da DBN (RBM) para o MLP (DeepNN)
        # Atenção: O peso W da RBM é [visible_size, hidden_size]
        # O peso weight da Linear (MLP) é [hidden_size, visible_size] -> Transposição necessária!
        
        # RBM 1 -> Layer 1
        self.layer1.weight.data.copy_(dbn_state_dict['rbm_layers.0.W'].data.T)
        self.layer1.bias.data.copy_(dbn_state_dict['rbm_layers.0.h_bias'].data)
        print("  > Pesos da RBM 1 carregados para a Camada 1 (16 -> 256)")
        
        # RBM 2 -> Layer 2
        self.layer2.weight.data.copy_(dbn_state_dict['rbm_layers.1.W'].data.T)
        self.layer2.bias.data.copy_(dbn_state_dict['rbm_layers.1.h_bias'].data)
        print("  > Pesos da RBM 2 carregados para a Camada 2 (256 -> 128)")
        
        # RBM 3 -> Layer 3
        self.layer3.weight.data.copy_(dbn_state_dict['rbm_layers.2.W'].data.T)
        self.layer3.bias.data.copy_(dbn_state_dict['rbm_layers.2.h_bias'].data)
        print("  > Pesos da RBM 3 carregados para a Camada 3 (128 -> 64)")
        
        # A Camada de Classificação (classifier_head) é deixada como inicializada (random)
        print("  > Camada de Classificação inicializada aleatoriamente.")
        print("✅ Carregamento de Pesos Concluído.")
