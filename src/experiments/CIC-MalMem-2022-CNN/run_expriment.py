import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import json
# Adiciona o diret�rio raiz do projeto ao path do Python para encontrar os m�dulos
# O script est� em src/experiments, a raiz � 2 n�veis acima.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)
# Importa��es universais
from src.modules.evaluation.evaluator import run_kfold_evaluation
from src.modules.utils.utils import EarlyStopping 
from src.models.CNN_CIC_MalMem.cnn_image import CNN_Image 
from src.dataloaders.CIC_MalMem.image_loader import MalMemImageDataset 

# ----------------------------------------------------------------------
# Caminhos e Fun��es de Carregamento
# ----------------------------------------------------------------------

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

def load_config():
    """Carrega o arquivo de configuração config.yaml."""
    config_path = PROJECT_ROOT / 'config.yaml'
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração não encontrado em: {config_path}")
        sys.exit(1)
        
def main():
    config = load_config()
    model_name = "CNN_Image_8x8" 
    
    # --- SETUP DO EXPERIMENTO ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(PROJECT_ROOT, 'results', model_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")
    shutil.copy(PROJECT_ROOT / 'config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    
    # 1. Definir os caminhos dos dados
    X_PATH = PROCESSED_DATA_DIR / 'X_image.npy' # Caminho para a representação 2D
    Y_PATH = PROCESSED_DATA_DIR / 'Y_final.npy'
    
    if not X_PATH.exists() or not Y_PATH.exists():
        print("ERRO: Arquivos de dados processados n�o encontrados. Execute data_processor.py.")
        return

    # 2. Execu��o da Avalia��o K-FOLD (A l�gica de Holdout e K-Fold � feita DENTRO do evaluator)
    results = run_kfold_evaluation(
        model_class=CNN_Image,
        model_name=model_name,
        config=config,
        X_data_path=X_PATH, # Caminho para o X_image.npy
        Y_data_path=Y_PATH, # Caminho para o Y_final.npy
        dataset_class=MalMemImageDataset, # A classe de Dataset que sabe ler X_image.npy
        experiment_dir=experiment_dir,
        early_stopper_class=EarlyStopping 
    )

    results_path = os.path.join(experiment_dir, 'summary_results.json')
    # Salvamos o resultado em JSON, que � mais comum para m�tricas
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4) 
    print(f"\nResultados de sumariza��o salvos em: {results_path}")

if __name__ == '__main__':
    main()