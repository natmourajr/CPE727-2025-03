import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

# Define o diretório raiz do projeto 
# O DataLoader está em src/dataloaders/CIC-MalMem-2022, precisa subir 3 níveis para a raiz
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

class MalMemImageDataset(Dataset):
    """
    Dataset customizado para PyTorch que carrega dados de imagem (2D)
    processados para a CNN.
    """
    def __init__(self, X_data, Y_data=None):
        """
        Inicializa o dataset.
        Args:
            X_data (np.ndarray): O array NumPy das features (X) no formato (N, H, W).
            Y_data (np.ndarray, opcional): O array NumPy dos rótulos (Y).
        """
        
        # 1. Converte NumPy para Tensor Float
        X_tensor = torch.FloatTensor(X_data)
        
        # 2. CRUCIAL PARA CNN: Adicionar a dimensão do canal (Channels).
        # A forma muda de (N, Altura, Largura) para (N, 1, Altura, Largura), 
        # onde 1 representa um canal (escala de cinza).
        self.X = X_tensor.unsqueeze(1) 
        
        if Y_data is not None:
            # Rótulos (Y) devem ser LongTensor para a função de perda CrossEntropyLoss
            self.Y = torch.LongTensor(Y_data)
            self.has_labels = True
        else:
            self.Y = None
            self.has_labels = False

    def __len__(self):
        """Retorna o número total de amostras."""
        return len(self.X)

    def __getitem__(self, idx):
        """Retorna a amostra e o rótulo para o índice (idx) fornecido."""
        if self.has_labels:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx]

# ----------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA PREPARAÇÃO DOS DATASETS (Holdout e K-Fold)
# ----------------------------------------------------------------------

def setup_kfold_dataloaders(X_data, Y_data, n_folds=5, test_size=0.2, batch_size=64, random_state=42):
    """
    Configura o Stratified K-Fold para treino/validação usando o MalMemImageDataset.
    (A lógica de separação é idêntica à do DataLoader Tabular).
    """
    
    print(f"--- Configurando separação de dados ---")
    
    # 1. Separação Holdout (Teste Final) - Estratificada
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_data, Y_data, 
        test_size=test_size, 
        stratify=Y_data, 
        random_state=random_state
    )
    print(f"Total de amostras: {len(X_data)}. Holdout (Teste Final): {len(X_test)}")
    print(f"Treino + Validação (K-Fold): {len(X_train_val)}")

    # Cria o DataLoader de Teste Final (Holdout)
    test_dataset = MalMemImageDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Configuração do Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Gerador para os folds de treino/validação
    def fold_dataloader_generator():
        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train_val, Y_train_val)):
            
            # 2.1. Obter os dados do Fold atual
            X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
            Y_train_fold, Y_val_fold = Y_train_val[train_index], Y_train_val[val_index]
            
            # 2.2. Criar Datasets e DataLoaders para Treino e Validação
            train_dataset = MalMemImageDataset(X_train_fold, Y_train_fold)
            val_dataset = MalMemImageDataset(X_val_fold, Y_val_fold)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            yield fold_idx, train_dataloader, val_dataloader

    return test_dataloader, fold_dataloader_generator()

# ----------------------------------------------------------------------
# EXEMPLO DE USO
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Carregar os arquivos processados (X_image para a CNN)
    try:
        X_data = np.load(PROCESSED_DATA_DIR / 'X_image.npy')
        Y_data = np.load(PROCESSED_DATA_DIR / 'Y_final.npy')
    except FileNotFoundError:
        print("ERRO: Arquivos .npy não encontrados. Execute data_processor.py primeiro.")
        exit()

    print(f"Dimensão de entrada bruta (NumPy): {X_data.shape}")
    
    # Exemplo de configuração e uso
    TEST_DL, FOLD_GENERATOR = setup_kfold_dataloaders(X_data, Y_data, n_folds=5, batch_size=128)
    
    print("\n--- Teste de Saída do DataLoader (Saída Esperada para CNN) ---")
    X_batch, Y_batch = next(iter(TEST_DL))
    
    # A forma final deve ser (Batch, Channels, Altura, Largura)
    print(f"Shape do batch de features (X): {X_batch.shape} (Batch, Canais, Altura, Largura)")
    print(f"Shape do batch de rótulos (Y): {Y_batch.shape} (Batch)")
    
    print("\nDataLoader de Imagem (CNN) configurado com sucesso!")
    