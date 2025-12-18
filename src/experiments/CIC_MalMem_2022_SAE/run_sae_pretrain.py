import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import os
import time
from pathlib import Path
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
import joblib 
from datetime import datetime
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
#sys.path.append(str(PROJECT_ROOT))
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

from src.models.Autoencoder_CIC_MalMem.autoencoder import Autoencoder 

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

def plot_loss_curves(train_history, save_dir, filename='loss_curves.png'):
    """Gera e salva o gráfico da curva de perda de treino."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Perda de Treino (MSE)')
    plt.title('Curva de Perda do Autoencoder')
    plt.xlabel('Épocas')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

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
    scaler = MinMaxScaler()
    X_data_scaled = scaler.fit_transform(X_data)
    # Assumindo que a forma de entrada é o número de features (16)
    input_size = X_data.shape[1] 
    
    # 2. Configurar DataLoader (usando todos os dados para pré-treino não supervisionado)
    # NÃO precisa de Holdout ou K-Fold aqui.
    pretrain_dataset = AEPretrainDataset(X_data_scaled)
    batch_size = config['training']['batch_size']
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Inicializar Modelo e Otimizador
    ae_config = config['architectures'].get('Autoencoder_SAE', {}) # Use a chave específica do AE no config
    
    model = Autoencoder(input_shape=input_size, num_classes=None, config=config).to(device)
    
    # A perda de reconstrução é Mean Squared Error (MSE)
    criterion = nn.MSELoss() 
    
    lr = config['training'].get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = ae_config.get('pretrain_epochs') # Reduzido para pré-treino (pode ser ajustado no config)

    print(f"\n--- INICIANDO PRÉ-TREINAMENTO DO {model_name} (Input: {input_size} features) ---")
    # Inicializa o histórico de perda
    train_loss_history = []
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
        train_loss_history.append(avg_loss)
        print(f"Época {epoch + 1}/{epochs} -> Perda de Reconstrução (MSE): {avg_loss:.6f}")

    end_time = time.time()
    
    # 4. Salvar Artefatos e Plotar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(PROJECT_ROOT, 'results', model_name, 'pretrain', timestamp) # Uso de timestamp para salvar
    os.makedirs(experiment_dir, exist_ok=True)

    # 5. Salvar os Pesos do Encoder
    encoder_weights_path = os.path.join(experiment_dir, 'best_encoder_weights.pt')
    os.makedirs(experiment_dir, exist_ok=True)
    
    plot_loss_curves(train_loss_history, experiment_dir, filename='ae_pretrain_loss_curve.png')

    # Salva o modelo completo (ou apenas o state_dict do encoder, dependendo da necessidade)
    torch.save(model.state_dict(), encoder_weights_path) 
    print(f"\n✅ Pré-treinamento concluído em {end_time - start_time:.2f}s.")
    print(f"Pesos do Encoder (e Decoder) salvos em: {encoder_weights_path}")
    
    # Retorna o caminho para os pesos
    return encoder_weights_path

if __name__ == '__main__':
    run_ae_pretrain()