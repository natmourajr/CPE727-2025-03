import yaml
import sys
import os
from pathlib import Path
import json
import shutil
# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importações de classes e funções
from src.modules.evaluation.evaluator_cic import run_kfold_evaluation
from src.modules.utils.utils import EarlyStopping 
from src.models.DeepNN_DBN_CIC_MalMem.deepnn_dbn import DeepNN_DBN_Classifier 
from src.dataloaders.CIC_MalMem.tabular_loader import MalMemTabularDataset
from src.experiments.CIC_MalMem_2022_DBN.run_dbn_pretrain import run_dbn_pretrain # Importa a função de pré-treino

# --- Funções Auxiliares de Caminho ---
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODEL_NAME = "CIC_MalMem_2022_DBN"

def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    # -----------------------------------------------------------
    # FASE 1: PRÉ-TREINAMENTO NÃO SUPERVISIONADO (JOB 3)
    # -----------------------------------------------------------
    print("\n--- INICIANDO JOB 3: PRÉ-TREINAMENTO NÃO SUPERVISIONADO (DBN) ---")
    
    # Esta função executa o treino da DBN (camada por camada) e SALVA os pesos.
    # A função deve retornar o caminho completo para o arquivo de pesos (.pt).
    dbn_weights_path = run_dbn_pretrain() 
    
    # -----------------------------------------------------------
    # FASE 2: FINE-TUNING SUPERVISIONADO K-FOLD (JOB 4)
    # -----------------------------------------------------------
    print("\n--- INICIANDO JOB 4: FINE-TUNING K-FOLD SUPERVISIONADO (DBN) ---")
    
    # 1. SETUP DE DIRETÓRIO
    # Usa o timestamp do pré-treino para agrupar o resultado.
    timestamp = Path(dbn_weights_path).parent.parent.name
    experiment_dir = os.path.join(RESULTS_DIR, MODEL_NAME, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy(PROJECT_ROOT / 'config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    print(f"Artefatos de Classificação DBN serão salvos em: {experiment_dir}")
    
    # 2. CAMINHOS DE DADOS (Features filtradas por ANOVA)
    X_PATH = PROCESSED_DATA_DIR / 'X_tabular_filtered.npy'
    Y_PATH = PROCESSED_DATA_DIR / 'Y_final.npy'
    
    # 3. EXECUÇÃO DO K-FOLD (Chamada à função universal)
    results = run_kfold_evaluation(
        model_class=DeepNN_DBN_Classifier, # Classificador que carrega os pesos DBN
        model_name=MODEL_NAME,
        config=config, 
        X_data_path=X_PATH, 
        Y_data_path=Y_PATH, 
        dataset_class=MalMemTabularDataset,
        experiment_dir=experiment_dir,
        early_stopper_class=EarlyStopping,
        pretrain_weights_path=dbn_weights_path # Argumento crucial para a transferência de pesos
    )
    
    # 4. SALVAR SUMÁRIO FINAL
    results_path = os.path.join(experiment_dir, 'final_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4) 
        
    print(f"\n✅ Avaliação K-Fold DBN Concluída. Resultados salvos.")
    return results

if __name__ == '__main__':
    # O orquestrador é chamado sem argumentos. Ele gerencia as chamadas internas.
    main()