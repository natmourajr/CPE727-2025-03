import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

# Define o diret�rio raiz do projeto para garantir que encontremos os dados
# O DataLoader est� em src/dataloaders/CIC-MalMem-2022, precisa subir 2 n�veis para a raiz
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

class MalMemTabularDataset(Dataset):
    """
    Dataset customizado para PyTorch que carrega dados tabulares (1D)
    processados para DeepNN e Autoencoder.
    """
    def __init__(self, X_data, Y_data=None):
        """
        Inicializa o dataset.
        Args:
            X_data (np.ndarray): O array NumPy das features (X).
            Y_data (np.ndarray, opcional): O array NumPy dos r�tulos (Y).
        """
        # Converte arrays NumPy para tensores PyTorch. 
        self.X = torch.FloatTensor(X_data)
        
        if Y_data is not None:
            # R�tulos (Y) devem ser LongTensor para a fun��o de perda CrossEntropyLoss
            self.Y = torch.LongTensor(Y_data)
            self.has_labels = True
        else:
            self.Y = None
            self.has_labels = False

    def __len__(self):
        """Retorna o n�mero total de amostras."""
        return len(self.X)

    def __getitem__(self, idx):
        """Retorna a amostra e o r�tulo para o �ndice (idx) fornecido."""
        if self.has_labels:
            return self.X[idx], self.Y[idx]
        else:
            # �til para o Autoencoder, que n�o precisa de r�tulo para o input
            return self.X[idx]

# ----------------------------------------------------------------------
# FUN��O AUXILIAR PARA PREPARA��O DOS DATASETS (Holdout e K-Fold)
# ----------------------------------------------------------------------

def setup_kfold_dataloaders(X_data, Y_data, n_folds=5, test_size=0.2, batch_size=64, random_state=42):
    """
    Carrega dados, separa o conjunto de holdout (teste final) e configura
    o Stratified K-Fold para treino/valida��o.
    """
    
    print(f"--- Configurando separa��o de dados ---")
    
    # 1. Separa��o Holdout (Teste Final) - Estratificada
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_data, Y_data, 
        test_size=test_size, 
        stratify=Y_data, 
        random_state=random_state
    )
    print(f"Total de amostras: {len(X_data)}. Holdout (Teste Final): {len(X_test)}")
    print(f"Treino + Valida��o (K-Fold): {len(X_train_val)}")

    # Cria o DataLoader de Teste Final (Holdout)
    test_dataset = MalMemTabularDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Configura��o do Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Gerador para os folds de treino/valida��o
    def fold_dataloader_generator():
        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train_val, Y_train_val)):
            
            # 2.1. Obter os dados do Fold atual
            X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
            Y_train_fold, Y_val_fold = Y_train_val[train_index], Y_train_val[val_index]
            
            # 2.2. Criar Datasets e DataLoaders para Treino e Valida��o
            train_dataset = MalMemTabularDataset(X_train_fold, Y_train_fold)
            val_dataset = MalMemTabularDataset(X_val_fold, Y_val_fold)
            
            # shuffle=True no Treino; shuffle=False na Valida��o/Teste
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            yield fold_idx, train_dataloader, val_dataloader

    return test_dataloader, fold_dataloader_generator()

# ----------------------------------------------------------------------
# EXEMPLO DE USO (apenas para teste inicial)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Carregar os arquivos processados (X_tabular para o DeepNN)
    try:
        X_data = np.load(PROCESSED_DATA_DIR / 'X_tabular.npy')
        Y_data = np.load(PROCESSED_DATA_DIR / 'Y_final.npy')
    except FileNotFoundError:
        print("ERRO: Arquivos .npy n�o encontrados. Execute data_processor.py primeiro.")
        exit()

    print(f"Dimens�o de entrada para o modelo: {X_data.shape[1]}")
    
    # Exemplo de configura��o e uso
    TEST_DL, FOLD_GENERATOR = setup_kfold_dataloaders(X_data, Y_data, n_folds=5, batch_size=128)
    
    print("\n--- Exemplo de uso do DataLoader de Teste (Holdout) ---")
    X_batch, Y_batch = next(iter(TEST_DL))
    print(f"Shape do batch de features (X): {X_batch.shape} (Batch, Features)")
    print(f"Shape do batch de r�tulos (Y): {Y_batch.shape} (Batch)")
    
    print("\nDataLoader Tabular configurado com sucesso!")