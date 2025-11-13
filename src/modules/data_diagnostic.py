import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr

# Configuração de Caminho (Ajuste o número de .parent() conforme sua estrutura)
PROJECT_ROOT = Path(__file__).parent.parent.parent 
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# O NOME DO SEU ARQUIVO RAW DEVE SER DEFINIDO AQUI
RAW_FILE_NAME = 'Obfuscated-MalMem2022.csv' 
RAW_DATA_PATH = RAW_DATA_DIR / RAW_FILE_NAME

def run_leakage_diagnosis(target_column='Class', correlation_threshold=0.99):
    """
    Identifica features com alta correlação com o rótulo (data leakage)
    que causam AUC = 1.0.
    """
    print(f"--- INICIANDO DIAGNÓSTICO DE VAZAMENTO DE DADOS (R > {correlation_threshold}) ---")
    
    try:
        # 1. Carregar dados brutos
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERRO: Arquivo bruto não encontrado em {RAW_DATA_PATH}. Execute data_processor.py.")
        return

    # 2. Pré-processamento e extração de X e Y (dados não escalonados)
    X = df_raw.drop(columns=[target_column, 'Category'], errors='ignore').fillna(0)
    Y = df_raw[target_column]

    # 3. Codificação do rótulo Y
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    # Lista para armazenar resultados
    leakage_features = []
    
    print(f"\nTotal de Features a Analisar: {X.shape[1]}")
    
    # 4. Cálculo da Correlação Feature-Target (Ponto-Bisserial para numérico X binário)
    # Se você tiver mais de 2 classes (Multiclasse), faremos uma simplificação.
    
    # NOTE: Usando a Correlação Ponto-Bisserial se binário, ou uma aproximação.
    # Vamos converter o Y codificado para o formato float (0.0 ou 1.0) para a correlação.
    
    if len(le.classes_) == 2:
        print("Usando Correlação Ponto-Bisserial (Problema Binário).")
        # Itera sobre cada coluna de feature
        for col in X.columns:
            # Garante que a feature seja numérica
            if np.issubdtype(X[col].dtype, np.number):
                try:
                    # Calcula a Correlação R e o p-valor
                    r, p_value = pointbiserialr(X[col].values, Y_encoded)
                    
                    if abs(r) >= correlation_threshold:
                        leakage_features.append((col, r))
                        
                except ValueError:
                    # Pode ocorrer se a feature tiver variação zero
                    pass
    else:
        # Se for Multiclasse, usaremos Informação Mútua (Mutual Information) como proxy
        print("Usando Mutual Information (MI) para Multiclasse (proxy para vazamento).")
        # MI não dá um R entre 0 e 1, mas uma pontuação de relevância.
        mi_scores = mutual_info_classif(X, Y_encoded, random_state=42)
        
        # Converte em Series para fácil análise
        mi_series = pd.Series(mi_scores, index=X.columns)
        
        # Filtra as features com pontuação de MI muito alta (indicando alta relevância)
        # O limiar aqui é heurístico, vamos usar a média + 2 desvios padrão
        mi_threshold = mi_series.mean() + 2 * mi_series.std()
        
        for col, mi_score in mi_series.items():
             if mi_score >= mi_threshold:
                 leakage_features.append((col, mi_score))


    print("\n--- FEATURES COM VAZAMENTO DETECTADO ---")
    if leakage_features:
        print(f"Encontradas {len(leakage_features)} features suspeitas (R > {correlation_threshold} ou MI Alto):")
        for col, score in leakage_features:
             print(f"  ❌ {col}: Score={score:.4f}")
             
        # Salvar a lista para usar no filtro
        leakage_cols = [col for col, score in leakage_features]
        np.save(PROCESSED_DATA_DIR / 'leakage_features.npy', leakage_cols)
        print(f"\n✅ Lista de colunas vazadas salva em: {PROCESSED_DATA_DIR / 'leakage_features.npy'}")
    else:
        print("✅ Nenhuma feature vazada encontrada acima do limiar. O problema pode ser outro.")
        
    return leakage_features

if __name__ == '__main__':
    # Necessita que o arquivo raw esteja em data/raw/
    run_leakage_diagnosis(target_column='Class')