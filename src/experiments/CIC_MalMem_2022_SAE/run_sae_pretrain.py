import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import os
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

from models.Autoencoder_CIC_MalMem.autoencoder import Autoencoder 

# ----------------------------------------------------------------------
# Dataset Customizado para AE (Não precisa de rótulo Y)
# ----------------------------------------------------------------------
class AEPretrainDataset(Dataset):
    def __init__(self, X_data):
        self.X = torch.FloatTensor(X_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retorna apenas a feature X, pois o AE a usa como target de reconstrução
        return self.X[idx] 

def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_ae_pretrain():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Parâmetros ---
    model_name = "Autoencoder_SAE"
    experiment_dir = os.path.join(PROJECT_ROOT, 'results', model_name, 'pretrain')
    os.makedirs(experiment_dir, exist_ok=True)

    # 1. Carregar Dados Tabulares Filtrados (16 Features)
    X_path = PROCESSED_DATA_DIR / 'X_tabular_filtered.npy'
    X_data = np.load(X_path)
    
    # Assumindo que a forma de entrada é o número de features (16)
    input_size = X_data.shape[1] 
    
    # 2. Configurar DataLoader (usando todos os dados para pré-treino não supervisionado)
    # NÃO precisamos de Holdout ou K-Fold aqui.
    pretrain_dataset = AEPretrainDataset(X_data)
    batch_size = config['training']['batch_size']
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Inicializar Modelo e Otimizador
    ae_config = config['architectures'].get('Autoencoder_SAE', {}) # Use a chave específica do AE no config
    
    model = Autoencoder(input_shape=input_size, num_classes=None, config=config).to(device)
    
    # A perda de reconstrução é Mean Squared Error (MSE)
    criterion = nn.MSELoss() 
    
    lr = config['training'].get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 50 # Reduzido para pré-treino (pode ser ajustado no config)

    print(f"\n--- INICIANDO PRÉ-TREINAMENTO DO {model_name} (Input: {input_size} features) ---")
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        
        for X_batch in pretrain_loader: # DataLoader retorna apenas X_batch
            X_batch = X_batch.to(device)
            
            # Forward: X -> Reconstrução
            reconstruction = model(X_batch)
            
            # Loss: Reconstrução vs. Entrada Original (X)
            loss = criterion(reconstruction, X_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_loss = train_loss_sum / len(pretrain_loader)
        print(f"Época {epoch + 1}/{epochs} -> Perda de Reconstrução (MSE): {avg_loss:.6f}")

    end_time = time.time()
    
    # 4. Salvar os Pesos do Encoder
    encoder_weights_path = os.path.join(experiment_dir, 'best_encoder_weights.pt')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Salva o modelo completo (ou apenas o state_dict do encoder, dependendo da necessidade)
    torch.save(model.state_dict(), encoder_weights_path) 
    print(f"\n✅ Pré-treinamento concluído em {end_time - start_time:.2f}s.")
    print(f"Pesos do Encoder (e Decoder) salvos em: {encoder_weights_path}")
    
    # Retorna o caminho para os pesos
    return encoder_weights_path

if __name__ == '__main__':
    # Você precisará rodar o data_processor.py antes!
    run_ae_pretrain()