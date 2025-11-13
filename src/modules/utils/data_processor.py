import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import os
import pickle
import joblib

# ----------------------------------------------------------------------
# CORREÇÃO DE PATHS: Encontrando o diretório raiz do projeto
# ----------------------------------------------------------------------
# O script está em src/modules/utils, precisamos subir 3 níveis.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
#print(PROJECT_ROOT)


# Define os caminhos absolutos
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# Garante que as pastas existam
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------

def _encode_labels(df, target_column='Class'):
    """Codifica a coluna target para valores inteiros (0, 1, 2...)."""
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    
    # Salva o LabelEncoder
    with open(PROCESSED_DATA_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print(f"  -> Rótulos de classe codificados. Encoder salvo em: {PROCESSED_DATA_DIR / 'label_encoder.pkl'}")
    return df, le

#def _scale_features(X):
#    """Escalona as features numéricas usando o StandardScaler."""
#    scaler = StandardScaler()
#    X_scaled = scaler.fit_transform(X)
#    
#    # Salva o Scaler
#    with open(PROCESSED_DATA_DIR / 'feature_scaler.pkl', 'wb') as f:
#        pickle.dump(scaler, f)
#        
#    print(f"  -> Features escalonadas (StandardScaler). Scaler salvo.")
#    return X_scaled

# ----------------------------------------------------------------------
# 1. TRANSFORMAÇÃO VETORIAL/TABULAR (DeepNN, AE)
# ----------------------------------------------------------------------

def process_tabular_data(df, target_column='Class'):
    """Prepara os dados no formato vetorial (tabular) para DeepNN e AE."""
    print("Iniciando processamento de dados TABULARES (Vetorial)...")
    
    X = df.drop(columns=[target_column, 'Category'], errors='ignore')
    Y = df[[target_column]]
    
    X = X.fillna(0)
    
    df_with_encoded_labels, _ = _encode_labels(df.copy(), target_column)
    Y_encoded = df_with_encoded_labels[target_column].values
    
    #X_scaled = _scale_features(X)
    
    # SALVAMENTO: Y_tabular.npy é redundante, mas mantido para evitar confusão de nomes de arquivos.
    np.save(PROCESSED_DATA_DIR / 'X_tabular.npy', X.astype(np.float32)) # Salva X BRUTO
    np.save(PROCESSED_DATA_DIR / 'Y_final.npy', y.astype(np.int64))
    
    # Salva apenas o LabelEncoder, pois o StandardScaler ser� criado no evaluator
    joblib.dump(label_encoder, PROCESSED_DATA_DIR / 'label_encoder.pkl')
    
    return X, y
# ----------------------------------------------------------------------
# 2. TRANSFORMAÇÃO GRÁFICA/2D (CNN)
# ----------------------------------------------------------------------

def process_image_data(X_scaled):
    """Transforma o vetor de features escalonadas em uma matriz 2D para CNN."""
    print("\nIniciando processamento de dados GRÁFICOS (2D/Imagem)...")
    
    N = X_scaled.shape[1] 
    side = int(np.ceil(np.sqrt(N)))
    total_pixels = side * side
    padding_needed = total_pixels - N
    
    X_padded = np.pad(X_scaled, ((0, 0), (0, padding_needed)), 'constant', constant_values=0)
    X_images = X_padded.reshape(-1, side, side)
    
    np.save(PROCESSED_DATA_DIR / 'X_image.npy', X_images)
    
    print(f"  -> Features 2D (Imagens) salvas: X_image.npy")
    print(f"  -> Dimensão de Imagem definida: ({side}x{side})")
    return 1, side, side

# ----------------------------------------------------------------------
# 3. TRANSFORMAÇÃO SEQUENCIAL (LSTM)
# ----------------------------------------------------------------------

def process_sequential_data(X_scaled):
    """Transforma features escalonadas em uma sequência de vetores para LSTM."""
    print("\nIniciando processamento de dados SEQUENCIAIS...")
    
    N = X_scaled.shape[1]
    SEQ_LEN = 9 
    FEAT_PER_STEP = N // SEQ_LEN 
    
    if N % SEQ_LEN != 0:
        print(f"  -> Aviso: N={N} não é divisível por {SEQ_LEN}. Usando {FEAT_PER_STEP * SEQ_LEN} features.")

    X_truncated = X_scaled[:, :FEAT_PER_STEP * SEQ_LEN]
    X_sequences = X_truncated.reshape(-1, SEQ_LEN, FEAT_PER_STEP)
    
    np.save(PROCESSED_DATA_DIR / 'X_sequential.npy', X_sequences)
    
    print(f"  -> Features Sequenciais salvas: X_sequential.npy")
    print(f"  -> Sequência definida: {SEQ_LEN} passos, {FEAT_PER_STEP} features/passo.")
    return SEQ_LEN, FEAT_PER_STEP

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------------------------------------------------

def run_data_processing(raw_file_name):
    """
    Executa todas as transformações de dados.
    Argumento é o nome do arquivo, que deve estar em data/raw/.
    """
    raw_data_path = RAW_DATA_DIR / raw_file_name
    print(f"--- INICIANDO PROCESSAMENTO DE DADOS DO CIC-MalMem-2022 ---")
    
    # 1. Carregar Dados Brutos
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {raw_data_path}. Verifique o caminho.")
        return
    
    # 2. Pré-Processamento Básico (Base Comum)
    X_scaled_tabular, Y_encoded = process_tabular_data(df)
    
    # 3. Transformações Específicas
    
    # 3.1. CNN (2D)
    process_image_data(X_scaled_tabular)
    
    # 3.2. LSTM (Sequencial)
    process_sequential_data(X_scaled_tabular)
    
    print("\n--- PROCESSAMENTO CONCLUÍDO. ARQUIVOS SALVOS EM data/processed/ ---")
    
    # Salvando Y final de forma consolidada.
    np.save(PROCESSED_DATA_DIR / 'Y_final.npy', Y_encoded)

if __name__ == '__main__':
    # EXECUTAR O SCRIPT A PARTIR DA RAIZ DO PROJETO.
    RAW_FILE_NAME = 'Obfuscated-MalMem2022.csv' 
    
    run_data_processing(RAW_FILE_NAME)