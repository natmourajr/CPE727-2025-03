import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from src.models.Autoencoder_CIC_MalMem.autoencoder import Autoencoder 
from src.models.DeepNN_CIC_MalMem.deepnn_mlp import DeepNN_MLP 

class DeepNN_SAE_Classifier(DeepNN_MLP):
    """
    Classificador DeepNN que utiliza pré-treinamento com Autoencoder (SAE).
    As camadas iniciais (Encoder) são inicializadas com pesos pré-treinados
    e o restante (Decodificador/Camada de Saída) é treinado do zero.
    """
    def __init__(self, input_shape, num_classes, config, pretrain_weights_path=None):
        
        # Inicializa a estrutura base do MLP (DeepNN_MLP)
        super(DeepNN_SAE_Classifier, self).__init__(input_shape, num_classes, config)
        
        # --- Lógica para Carregar Pesos do Encoder ---
        if pretrain_weights_path:
            self.load_pretrain_weights(pretrain_weights_path)
        
        self.fine_tuning = pretrain_weights_path is not None # Sinaliza que está em modo fine-tuning

    def load_pretrain_weights(self, pretrain_weights_path):
        """
        Carrega os pesos do Encoder do Autoencoder para as camadas do classificador (SAE).
        Assume a estrutura unificada: enc1->fc1, enc2->fc2, enc_code->fc3 e inclui camadas BN.
        """
        if not os.path.exists(pretrain_weights_path):
            print(f"AVISO: Arquivo de pesos pré-treinados não encontrado em {pretrain_weights_path}.")
            return

        print(f"Carregando pesos pré-treinados do Encoder do AE...")
        
        # Carrega o dicionário de estados
        ae_state_dict = torch.load(pretrain_weights_path)
        
        # --- Mapeamento e Transferência de Pesos (SAE) ---
        # Estrutura assumida:
        # AE Encoder: enc1 (Linear), enc2 (Linear), enc_code (Linear)
        # MLP Classifier: fc1, bn1, fc2, bn2, fc3, bn3, fc_out
        
        # 1. Mapeamento da Camada FC1
        self.fc1.weight.data.copy_(ae_state_dict['enc1.weight'].data)
        self.fc1.bias.data.copy_(ae_state_dict['enc1.bias'].data)
        
        # 2. Mapeamento dos Pesos da Camada BN1 (BN's precisam de seus pesos copiados)
        self.bn1.weight.data.copy_(ae_state_dict['bn1.weight'].data) # Corrigir se o AE não tiver BN
        self.bn1.bias.data.copy_(ae_state_dict['bn1.bias'].data)
        self.bn1.running_mean.copy_(ae_state_dict['bn1.running_mean'])
        self.bn1.running_var.copy_(ae_state_dict['bn1.running_var'])
        
        # 3. Mapeamento da Camada FC2
        self.fc2.weight.data.copy_(ae_state_dict['enc2.weight'].data)
        self.fc2.bias.data.copy_(ae_state_dict['enc2.bias'].data)

        # 4. Mapeamento da Camada Latente para FC3
        # Mapeia a camada de código (enc_code) do AE para a terceira camada linear (fc3) do MLP.
        self.fc3.weight.data.copy_(ae_state_dict['enc_code.weight'].data)
        self.fc3.bias.data.copy_(ae_state_dict['enc_code.bias'].data)
               
        print("Pesos do Encoder (incluindo BN) transferidos com sucesso para o classificador.")