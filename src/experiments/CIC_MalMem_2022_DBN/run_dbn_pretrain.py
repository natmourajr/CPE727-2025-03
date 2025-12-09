import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import os
import time
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 
import joblib 
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

from src.models.DBN_CIC_MalMem.dbn import DeepBeliefNetwork
from src.dataloaders.CIC_MalMem.tabular_loader import MalMemTabularDataset

MODEL_NAME = "CIC_MalMem_2022_DBN_RBM_PRETRAIN"

def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def plot_dbn_loss_curves(loss_history_list, save_dir, filename='dbn_loss_curves.png'):
    """Gera e salva o gráfico da curva de perda para cada RBM."""
    
    plt.figure(figsize=(12, 8))
    
    for i, history in enumerate(loss_history_list):
        plt.plot(history, label=f'RBM {i+1} Loss ({len(history)} epochs)')
        
    plt.title('Curvas de Perda do Pré-Treinamento DBN (Greedy Layer-wise)')
    plt.xlabel('Épocas')
    plt.ylabel('Perda de Reconstrução')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def run_dbn_pretrain():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SETUP DE DADOS
    X_PATH = PROCESSED_DATA_DIR / 'X_tabular_filtered.npy'
    Y_PATH = PROCESSED_DATA_DIR / 'Y_final.npy'
    
    X_data = np.load(X_PATH)
    Y_data = np.load(Y_PATH)
    scaler = MinMaxScaler()
    X_data_scaled = scaler.fit_transform(X_data)
    
    dataset = MalMemTabularDataset(X_data_scaled, Y_data)
    dataloader = DataLoader(dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True)

    # 2. SETUP DA ARQUITETURA
    input_size = 16 
    num_classes = 4 
    
    dbn_config = config['architectures'].get(MODEL_NAME, {})
    pretrain_epochs = dbn_config.get('pretrain_epochs', 50) 
    learning_rate = dbn_config.get('pretrain_lr', 0.01)

    print(f"\n--- INICIANDO PRÉ-TREINAMENTO DA {MODEL_NAME} (Input: {input_size} features) ---")
    dbn_model = DeepBeliefNetwork(input_size, num_classes, config, device=device)
    
    start_time = time.time()
    
    # 3. EXECUTAR PRÉ-TREINAMENTO DBN E CAPTURAR HISTÓRICO
    # A variável 'dbn_loss_history' deve ser capturada aqui.
    dbn_loss_history = dbn_model.pretrain(dataloader, pretrain_epochs, learning_rate)
    
    end_time = time.time()
    print(f"✅ Pré-treinamento concluído em {end_time - start_time:.2f}s.")

    # 4. SALVAR PESOS DO ENCODER (A lógica de salvamento está aqui)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RESULTS_DIR / MODEL_NAME / 'pretrain' / timestamp
    os.makedirs(save_dir, exist_ok=True)
    
    weights_path = save_dir / 'dbn_pretrain_weights.pt'
    torch.save(dbn_model.state_dict(), weights_path)
    
    print(f"Pesos do DBN salvos em: {weights_path}")

    # 5. PLOTAR HISTÓRICO 
    # dbn_loss_history é o retorno de dbn.pretrain(), assumido ser a lista de listas.
    if isinstance(dbn_loss_history, list) and len(dbn_loss_history) > 0 and isinstance(dbn_loss_history[0], list):
        plot_dbn_loss_curves(dbn_loss_history, save_dir)
    else:
        # Se dbn.pretrain() não foi corrigido para retornar a lista de listas,
        # ele cairá aqui e mostrará o AVISO (o que é intencional para debug).
        print("AVISO: Histórico de perda da DBN não foi capturado para plotagem.")

    return str(weights_path)

if __name__ == '__main__':
    run_dbn_pretrain()