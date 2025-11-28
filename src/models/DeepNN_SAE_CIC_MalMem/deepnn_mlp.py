import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.Autoencoder_CIC_MalMem.autoencoder import Autoencoder 
from models.DeepNN_CIC_MalMem.deepnn_mlp import DeepNN_MLP 

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
        Carrega os pesos do Encoder do Autoencoder para as camadas do classificador.
        """
        if not os.path.exists(pretrain_weights_path):
            print(f"AVISO: Arquivo de pesos pré-treinados não encontrado em {pretrain_weights_path}.")
            return

        print(f"Carregando pesos pré-treinados do Encoder do AE...")
        
        # Criar uma instância dummy do Autoencoder para carregar os pesos
        # Assumimos que o AE foi treinado com o mesmo input_size
        input_size = self.input_size if hasattr(self, 'input_size') else self.input_shape
        
        # NOTA: Aqui, você precisaria de uma forma mais robusta de instanciar o AE,
        # mas faremos um bypass para carregar o state_dict diretamente.
        
        ae_state_dict = torch.load(pretrain_weights_path)
        
        # Mapeamento e Transferência de Pesos (Crucial: Mapear enc1 -> fc1, enc2 -> fc2, etc.)
        # Assumindo que o MLP e o Encoder do AE têm as mesmas camadas iniciais:
        
        # Mapeamento do Encoder:
        self.fc1.weight.data.copy_(ae_state_dict['enc1.weight'].data)
        self.fc1.bias.data.copy_(ae_state_dict['enc1.bias'].data)
        self.fc2.weight.data.copy_(ae_state_dict['enc2.weight'].data)
        self.fc2.bias.data.copy_(ae_state_dict['enc2.bias'].data)

        # Se o seu MLP tem uma terceira camada de FC3 e o AE tem enc_code:
        # self.fc3.weight.data.copy_(ae_state_dict['enc_code.weight'].data)
        # self.fc3.bias.data.copy_(ae_state_dict['enc_code.bias'].data)
        
        # Opcional: Congelar as camadas pré-treinadas no início do fine-tuning
        # for param in [self.fc1.parameters(), self.fc2.parameters()]:
        #     for p in param:
        #         p.requires_grad = False
        
        print("Pesos do Encoder transferidos com sucesso para o classificador.")