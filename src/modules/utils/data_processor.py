import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import os
import pickle
import joblib

# ----------------------------------------------------------------------
# CORREÇÃO DE PATHS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------

def map_to_four_classes(category_name):
    """Mapeia as subfamílias de malware para as 4 categorias principais."""
    if category_name == 'Benign':
        return 'Benign'
    elif 'Trojan' in category_name or 'trojan' in category_name:
        return 'Trojan'
    elif 'Ransomware' in category_name or 'ransomware' in category_name:
        return 'Ransomware'
    elif 'Spyware' in category_name or 'spyware' in category_name:
        return 'Spyware'
    # Caso para qualquer outra entrada que não se encaixe (deve ser rara)
    else:
        return 'Other' 

def _encode_labels(df, target_column='Category_4Classes'):
    """Codifica a coluna target para valores inteiros (0, 1, 2...)."""
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    
    with open(PROCESSED_DATA_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print(f"  -> Rótulos de classe codificados. Encoder salvo.")
    return df, le

# ----------------------------------------------------------------------
# 1. TRANSFORMAÇÃO VETORIAL/TABULAR COM FEATURE SELECTION (DeepNN, AE)
# ----------------------------------------------------------------------

def process_tabular_data(df, target_column_name='Category', k_best=16):
    
    # 1. Agrupamento de Classes (De 17 para 4)
    df['Category_4Classes'] = df[target_column_name].apply(map_to_four_classes)
    TARGET_COLUMN_4 = 'Category_4Classes'
    
    print(f"Iniciando processamento TABULAR ({k_best} Features | Target: {TARGET_COLUMN_4})...")
    
    # 1.1. Separação inicial e remoção de targets
    # X deve ser um DataFrame temporário limpo.
    X = df.drop(columns=[target_column_name, 'Class', TARGET_COLUMN_4], errors='ignore')
    Y_4 = df[TARGET_COLUMN_4]
    
    X = X.fillna(0)
    
    # --- CORREÇÃO CRÍTICA: Filtrar colunas não numéricas em X ---
    # Manter APENAS as colunas que são de tipos numéricos (int ou float)
    X_numeric = X.select_dtypes(include=np.number)
    # -------------------------------------------------------------
    
    # 2. Codificação de Rótulos (Y)
    df_with_encoded_labels, label_encoder = _encode_labels(df.copy(), TARGET_COLUMN_4)
    Y_encoded = df_with_encoded_labels[TARGET_COLUMN_4].values
    
    # 3. Seleção de Features (ANOVA F-test)
    try:
        selector = SelectKBest(f_classif, k=k_best)
        # Aplicar o fit no conjunto X_numeric
        selector.fit(X_numeric, Y_encoded) 
        
        selected_features_mask = selector.get_support()
        X_selected = X_numeric.iloc[:, selected_features_mask] # Aplicar a máscara em X_numeric
        
        print(f"  -> Seleção ANOVA concluída. {X_selected.shape[1]} de {X_numeric.shape[1]} features numéricas selecionadas.")
        
    except ValueError as e:
        print(f"  ERRO: Falha na Seleção KBest ({e}). Usando todas as features numéricas.")
        X_selected = X_numeric # Fallback para todas as 55 features numéricas

    # SALVAMENTO: X filtrado (garantidamente numérico)
    # Use .values para converter o DataFrame em array NumPy antes de salvar
    np.save(PROCESSED_DATA_DIR / 'X_tabular_filtered.npy', X_selected.values.astype(np.float32))
    
    # Retorna o X FILTRADO (array NumPy) e o Y codificado
    return X_selected.values, Y_encoded
# ----------------------------------------------------------------------
# 2. TRANSFORMAÇÃO GRÁFICA/2D (CNN) - USA O CONJUNTO COMPLETO DE FEATURES
# ----------------------------------------------------------------------

def process_image_data(X_full):
    """
    Transforma o vetor de features COMPLETAS (55 features) em uma matriz 2D para CNN.
    X_full aqui é o DataFrame de todas as 55 features.
    """
    print("\nIniciando processamento de dados GRÁFICOS (2D/Imagem - 55 Features)...")
    
    # Certificar-se de que X_full é um array numpy para o shape
    X_full_array = X_full if isinstance(X_full, np.ndarray) else X_full.values
    
    N = X_full_array.shape[1] 
    side = int(np.ceil(np.sqrt(N))) # Deve resultar em 8x8 para 55 features
    total_pixels = side * side
    padding_needed = total_pixels - N
    
    X_padded = np.pad(X_full_array, ((0, 0), (0, padding_needed)), 'constant', constant_values=0)
    
    # --- CORREÇÃO AQUI: A linha de reshape estava faltando! ---
    X_images = X_padded.reshape(-1, side, side)
    
    # SALVAMENTO: X completo para CNN
    np.save(PROCESSED_DATA_DIR / 'X_image_full.npy', X_images.astype(np.float32))
    
    print(f"  -> Features 2D (Imagens) salvas: X_image_full.npy")
    print(f"  -> Dimensão de Imagem definida: ({side}x{side})")
    return X_images.astype(np.float32)

# ----------------------------------------------------------------------
# 3. TRANSFORMAÇÃO SEQUENCIAL (LSTM) - USA O CONJUNTO COMPLETO DE FEATURES
# ----------------------------------------------------------------------
def process_sequential_data(X_full):
    """
    Transforma features COMPLETAS (55 features) em uma sequência de vetores para LSTM.
    """
    print("\nIniciando processamento de dados SEQUENCIAIS (55 Features)...")
    
    # Garantir que X_full é um array numpy
    X_full_array = X_full if isinstance(X_full, np.ndarray) else X_full.values
    
    N = X_full_array.shape[1]
    SEQ_LEN = 9 
    FEAT_PER_STEP = N // SEQ_LEN # 55 // 9 = 6
    
    if N % SEQ_LEN != 0:
        print(f"  -> Aviso: N={N} não é divisível por {SEQ_LEN}. Usando {FEAT_PER_STEP * SEQ_LEN} features.")

    # Truncar o array para ser divisível por SEQ_LEN * FEAT_PER_STEP (ou seja, 54 colunas)
    X_truncated = X_full_array[:, :FEAT_PER_STEP * SEQ_LEN]
    
    # --- CORREÇÃO AQUI: Definir X_sequences ---
    # Reshape para (num_samples, seq_len, features_por_passo)
    X_sequences = X_truncated.reshape(-1, SEQ_LEN, FEAT_PER_STEP)
    
    # SALVAMENTO: X completo para LSTM
    np.save(PROCESSED_DATA_DIR / 'X_sequential_full.npy', X_sequences.astype(np.float32))
    
    print(f"  -> Features Sequenciais salvas: X_sequential_full.npy")
    print(f"  -> Sequência definida: {SEQ_LEN} passos, {FEAT_PER_STEP} features/passo.")
    return X_sequences.astype(np.float32) # <--- Retorna X_sequences

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------------------------------------------------

def run_data_processing(raw_file_name):
    """
    Executa todas as transformações de dados.
    """
    raw_data_path = RAW_DATA_DIR / raw_file_name
    print(f"--- INICIANDO PROCESSAMENTO DE DADOS DO CIC-MalMem-2022 (4 CLASSES) ---")
    
    # 1. Carregar Dados Brutos (CORREÇÃO: Adicionar o bloco de carregamento aqui)
    try:
        df = pd.read_csv(raw_data_path) # <--- ONDE DF É DEFINIDO
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {raw_data_path}. Verifique o caminho.")
        return
    
    # 2. Pré-Processamento Básico (Target: 'Category')
    # X_filtered (16 features) e Y_encoded (4 classes) são criados aqui.
    X_filtered, Y_encoded = process_tabular_data(df, target_column_name='Category', k_best=16)
    
    # 3. Gerar os demais conjuntos usando o X original (55 features)
    
    # 3.0. CRIAÇÃO DO DATAFRAME COMPLETO E LIMPEZA DE STRINGS
    # Usamos o DataFrame original 'df' e removemos apenas as colunas de rótulo para obter o X full.
    X_full_df = df.drop(columns=['Category', 'Class'], errors='ignore').fillna(0)
    
    # FILTRAR X_full_df para garantir que seja totalmente numérico (CORREÇÃO FINAL DE VALUERROR)
    X_full_df = X_full_df.select_dtypes(include=np.number)
    
    # 3.1. CNN (2D) - Usa as 55 features
    process_image_data(X_full_df)
    
    # 3.2. LSTM (Sequencial) - Usa as 55 features
    process_sequential_data(X_full_df)
    
    print("\n--- PROCESSAMENTO CONCLUÍDO. ARQUIVOS SALVOS EM data/processed/ ---")
    
    # Salvando Y final de forma consolidada.
    np.save(PROCESSED_DATA_DIR / 'Y_final.npy', Y_encoded)

if __name__ == '__main__':
    # EXECUTAR O SCRIPT A PARTIR DA RAIZ DO PROJETO.
    RAW_FILE_NAME = 'Obfuscated-MalMem2022.csv' 
    
    run_data_processing(RAW_FILE_NAME)