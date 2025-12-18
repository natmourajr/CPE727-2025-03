import yaml
import sys
import os
from pathlib import Path
import json
from datetime import datetime
# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importações de classes e funções
from src.modules.evaluation.evaluator_cic import run_kfold_evaluation
from src.modules.utils.utils import EarlyStopping 
from src.models.DeepNN_SAE_CIC_MalMem.deepnn_sae_classifier import DeepNN_SAE_Classifier 
from src.dataloaders.CIC_MalMem.tabular_loader import MalMemTabularDataset
from src.experiments.CIC_MalMem_2022_SAE.run_sae_pretrain import run_ae_pretrain # Importa a função de pré-treino

# --- Funções Auxiliares de Caminho ---
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    model_name = "CIC_MalMem_2022_DeepNN_SAE" # Nome para o experimento
    
    # --- 1. PRÉ-TREINAMENTO (JOB 1) ---
    pretrain_weights_path = run_ae_pretrain() # Executa o pré-treinamento
    
    # --- SETUP DO EXPERIMENTO SUPERVISIONADO (JOB 2) ---pretrain
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Usa o timestamp do pré-treino para agrupar
    experiment_dir = os.path.join(PROJECT_ROOT, 'results', model_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"\nArtefatos de Classificação SAE serão salvos em: {experiment_dir}")
    
    # 2. Definir os caminhos dos dados (FILTRADOS)
    X_PATH = PROCESSED_DATA_DIR / 'X_tabular_filtered.npy'
    Y_PATH = PROCESSED_DATA_DIR / 'Y_final.npy'
    
    # 3. Execução da Avaliação K-FOLD (Treinamento Supervisionado)
    results = run_kfold_evaluation(
        model_class=DeepNN_SAE_Classifier,
        model_name=model_name,
        config=config,
        X_data_path=X_PATH,
        Y_data_path=Y_PATH,
        dataset_class=MalMemTabularDataset,
        experiment_dir=experiment_dir,
        early_stopper_class=EarlyStopping,
        pretrain_weights_path=pretrain_weights_path 
    )

    # ... (restante do salvamento dos resultados) ...
    print("\nProcesso de avaliação SAE finalizado.")

if __name__ == '__main__':
    main()