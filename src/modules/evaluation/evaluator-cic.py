import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import shutil
from pathlib import Path
import json
import joblib

# Importa o DataLoader Tabular (assumindo que será usado para a maior parte dos experimentos)
# O script de experimento precisará importar o DataLoader correto (Tabular, Image ou Sequential)
from src.dataloaders.CIC_MalMem.tabular_loader import MalMemTabularDataset, setup_kfold_dataloaders # Usando o DataLoader que criamos
from src.dataloaders.CIC_MalMem.image_loader import MalMemImageDataset, setup_kfold_dataloaders # Usando o DataLoader que criamos
# ----------------------------------------------------------------------
# Configuração de Caminho
# ----------------------------------------------------------------------
# Ajuste o número de '.parent' se a estrutura de diretórios mudar
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True) 
print(PROJECT_ROOT)

# ----------------------------------------------------------------------
# Funções de Métricas e Plotagem
# ----------------------------------------------------------------------

def plot_roc_curve(y_true, y_probs, fold_name, set_name, save_dir):
    """Gera e salva o gr�fico da Curva ROC."""
    if len(np.unique(y_true)) < 2:
        # ... (c�digo)
        return

    # --- CORRE��O DE DIMENSIONALIDADE PARA CASO BIN�RIO ---
    if y_probs.ndim > 1 and y_probs.shape[1] == 2:
        y_probs_binary = y_probs[:, 1]
    else:
        y_probs_binary = y_probs
        
    fpr, tpr, _ = roc_curve(y_true, y_probs_binary)
    
    # O c�lculo do AUC deve usar o array completo (para OVR) se for multi-classe
    if y_probs.ndim > 1 and y_probs.shape[1] > 2:
        auc = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true, y_probs_binary)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
    plt.title(f'Curva ROC - {fold_name} - Conjunto de {set_name}')
    plt.legend(loc="lower right")
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'roc_curve_{set_name}.png'))
    plt.close()


def save_roc_curve_vectors(y_true, y_probs, set_name, save_dir):
    """
    Salva os vetores fpr, tpr e auc para plotagem comparativa posterior.
    Retorna a pontuação AUC.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
        
    # --- CORREÇÃO DE DIMENSIONALIDADE PARA CASO BINÁRIO ---
    # Se y_probs tem 2 colunas, é um problema binário (Multiclasse OVR não se aplica aqui)
    if y_probs.ndim > 1 and y_probs.shape[1] == 2:
        # Pega APENAS a probabilidade da classe positiva (Classe 1) para o roc_curve
        y_probs_binary = y_probs[:, 1]
    else:
        # Mantém o array original (para Multi-Classe real)
        y_probs_binary = y_probs
        
    # O roc_curve SÓ aceita 1D (para o caso binário)
    fpr, tpr, _ = roc_curve(y_true, y_probs_binary) 
    
    # O cálculo do AUC deve usar o array completo (para OVR)
    if y_probs.ndim > 1 and y_probs.shape[1] > 2:
        # Se for Multiclasse real
        auc = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
    else:
        # Se for Binário (2 classes), usa o cálculo binário normal
        auc = roc_auc_score(y_true, y_probs_binary) 

    roc_data_path = os.path.join(save_dir, f'roc_vectors_{set_name}.npz')
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, auc=auc)
    print(f"  -> Vetores da Curva ROC para '{set_name}' salvos.")
    return auc
    
def find_optimal_threshold(y_true, y_probs):
    """
    Encontra o limiar de decis�o �timo usando .
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Ignora o primeiro threshold que pode ser > 1
    sp = np.sqrt(  np.sqrt(tpr*(1-fpr)) * (0.5*(tpr+(1-fpr)))  )
    knee = np.argmax(sp)
    optimal_threshold = thresholds[knee]  
    
    return optimal_threshold   

def plot_loss_curves(train_history, val_history, fold, save_dir):
    """Gera e salva o gráfico das curvas de perda de treino e validação."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Perda de Treino')
    plt.plot(val_history, label='Perda de Validação')
    plt.title(f'Curvas de Perda - Fold {fold}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    

# Adicione esta função auxiliar no topo do evaluator.py, ou dentro da função, antes do loop:
def _scale_data_safely(X_raw, scaler, fit=True):
    """Aplica o StandardScaler com reshape se os dados forem 3D (CNN)."""
    
    # Se os dados são 3D (Imagens: N, H, W), achate para 2D (N, H*W)
    if X_raw.ndim == 3:
        # Armazena a forma original para remodelar mais tarde
        original_shape = X_raw.shape
        X_flat = X_raw.reshape(original_shape[0], -1) 
    else:
        # Se for 2D (Tabular), use diretamente
        X_flat = X_raw
        
    if fit:
        X_scaled_flat = scaler.fit_transform(X_flat)
    else:
        X_scaled_flat = scaler.transform(X_flat)
        
    # Remodela de volta para 3D se a entrada era 3D
    if X_raw.ndim == 3:
        return X_scaled_flat.reshape(original_shape)
    else:
        return X_scaled_flat


# ----------------------------------------------------------------------
# LÓGICA PRINCIPAL DE TREINAMENTO E AVALIAÇÃO K-FOLD
# ----------------------------------------------------------------------

# Classe Dummy para metadados (já que o DataLoader não retorna mais metadados)
class DummyMetadataBatch:
    def __init__(self, labels):
        self.labels = labels
        # Simula o dicionário de metadados anterior
        self.label = labels 
        
def run_kfold_evaluation(model_class, model_name, config, X_data_path, Y_data_path, dataset_class, experiment_dir, criterion=None, early_stopper_class=None):
    """
    Executa o treinamento e avaliação Stratified K-Fold para o modelo.
    """
    print(f"\n===== INICIANDO AVALIAÇÃO K-FOLD PARA O MODELO: {model_name} =====")
    
    # --- 1. Carregamento e Separação de Dados ---
    X_data = np.load(X_data_path) # Agora X_data está em escala BRUTA
    Y_data = np.load(Y_data_path)
    num_classes = len(np.unique(Y_data))
    
    # Define o DataLoader baseado na classe fornecida (Tabular, Image, etc.)
    dataloader_setup = setup_kfold_dataloaders
    
    # 1.1. Separação Holdout (20%)
    X_dev, X_test_raw, Y_dev, Y_test = train_test_split( # RENOMEADO X_test para X_test_raw (BRUTO)
        X_data, Y_data, test_size=config['cross_validation']['test_size'], stratify=Y_data, 
        random_state=config['dataset']['random_seed']
    )
    
    # Define a função de separação K-Fold para os dados DEV
    k_folds = config['cross_validation']['n_splits']
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    # Prepara o DataLoader do Holdout (TEMPORÁRIO, será refeito após carregamento do scaler)
    # OBS: O código anterior tinha um erro aqui, pois tentava criar o loader sem o scaler
    # Removido a linha temporária de holdout_loader para evitar confusão.
    
    print(f"Total de classes: {num_classes}. Holdout (Teste Final): {len(X_test_raw)} amostras.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")

    fold_results = []
    
    # --- 2. Loop de K-Fold ---
    for fold, (train_relative_indices, val_relative_indices) in enumerate(kf.split(X_dev, Y_dev)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{k_folds} ---")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Obter os dados do Fold atual (RAW)
        X_train_fold_raw, X_val_fold_raw = X_dev[train_relative_indices], X_dev[val_relative_indices]
        Y_train_fold, Y_val_fold = Y_dev[train_relative_indices], Y_dev[val_relative_indices]

        scaler = StandardScaler()

        # 1. Fit (aprender) o scaler APENAS nos dados de TREINO
        # CHAME A FUNÇÃO AUXILIAR AQUI:
        X_train_fold = _scale_data_safely(X_train_fold_raw, scaler, fit=True) 

        # 2. Transformar os dados de VALIDAÇÃO usando as estatísticas do Treino
        # CHAME A FUNÇÃO AUXILIAR AQUI:
        X_val_fold = _scale_data_safely(X_val_fold_raw, scaler, fit=False)
        # SALVAR O SCALER DO FOLD (Necessário para a avaliação final do Holdout)
        import joblib # Assumindo que este import está no topo do seu arquivo
        joblib.dump(scaler, os.path.join(fold_dir, 'fold_scaler.pkl'))

        # Criar DataLoaders (usando os dados ESCALONADOS)
        train_dataset_fold = dataset_class(X_train_fold, Y_train_fold)
        val_dataset_fold = dataset_class(X_val_fold, Y_val_fold)
        train_loader = DataLoader(train_dataset_fold, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=config['training']['batch_size'], shuffle=False)

        # Inicialização do Modelo
        # ... (Resto do Loop de K-Fold sem alterações) ...
        if X_data.ndim == 2:
            input_shape = X_data.shape[1] # Vetorial (Features)
        elif X_data.ndim == 4: # 4D (N, C, H, W) é o que a CNN espera, mas o NumPy é (N, H, W) (3D)
            input_shape = X_data.shape[1:] # (H, W) para a CNN
        else:
        # Captura a forma do array NumPy 3D (N, H, W)
            input_shape = X_data.shape[1:] 

        model = model_class(input_shape=input_shape, # Passa (55) ou (8, 8)
                            num_classes=num_classes, 
                            config=config).to(device)
        
        # Otimizador e Critério de Perda
        lr = config['training'].get('learning_rate', 0.001)
        wd = float(config['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Early Stopping (Assumindo que você tem essa classe em modules.Utils.utils)
        # Note: Esta linha requer que você implemente a classe EarlyStopping separadamente.
        early_stopper = early_stopper_class(patience=config['training'].get('early_stopping_patience', 7), 
                                            verbose=True, 
                                            path=os.path.join(fold_dir, 'best_model.pt'))
        
        train_loss_history, val_loss_history = [], []
        
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        # Loop de treinamento
        epochs = config['training']['epochs']
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            train_iterator = tqdm(train_loader, desc=f"  Treino Época {epoch + 1}/{epochs}", unit="batch")
            
            for X_batch, Y_batch in train_iterator: # Corrigido para a saída do DataLoader
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                # Lógica simplificada para classificação
                y_pred = model(X_batch)
                loss = criterion(y_pred, Y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            
            # Validação
            model.eval()
            val_loss_sum = 0.0
            val_iterator = tqdm(val_loader, desc=f"  Validação Época {epoch + 1}/{epochs}", unit="batch")
            with torch.no_grad():
                for X_batch, Y_batch in val_iterator: # Corrigido para a saída do DataLoader
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    val_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)
            avg_val_loss = val_loss_sum / len(val_loader)
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            print(f"  Época {epoch + 1}/{epochs} -> Perda Treino: {avg_train_loss:.6f} | Perda Validação: {avg_val_loss:.6f}")

            early_stopper(avg_val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping ativado!")
                break
            
        end_time = time.time()
        
        plot_loss_curves(train_loss_history, val_loss_history, fold_num, fold_dir)
        print("Carregando o melhor modelo salvo para avaliação final do fold...")
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
        
        fold_metrics = {
            'tempo_treino_seg': end_time - start_time,
            'pico_memoria_mb': torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        }

        # --- 3. Avaliação no Conjunto de Validação (Para encontrar o Limiar Ótimo) ---
        
        all_y_true, all_probs = [], []
        model.eval()
        with torch.no_grad():
            iterator = tqdm(val_loader, desc=f"  Avaliação Final Validação", unit="batch")
            for X_batch, Y_batch in iterator:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                
                # Softmax para obter probabilidades e pegar a probabilidade das classes (Multiclasse)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                
                all_probs.extend(probabilities)
                all_y_true.extend(Y_batch.cpu().numpy())
        
        val_y_true = np.array(all_y_true)
        val_y_probs = np.array(all_probs)
        
        # Encontra o limiar ótimo no conjunto de VALIDAÇÃO (Usado para todas as métricas de classificação)
        if len(np.unique(val_y_true)) > 1:
            # Necessário para ROC_AUC Multi-Classe OVR. Não é o melhor para limiar, mas vamos manter a função
            # Para manter a compatibilidade, vamos pegar a prob da primeira classe vs resto.
            optimal_threshold_for_fold = find_optimal_threshold(val_y_true, val_y_probs[:, 1])
            fold_metrics['optimal_threshold'] = float(optimal_threshold_for_fold)


            if num_classes == 2:
                # Para AUC binário, usamos APENAS a coluna de probabilidade da Classe 1
                val_y_probs_auc = val_y_probs[:, 1]
                # O ROC-AUC binário não precisa dos argumentos average/multi_class
                auc_val_geral = roc_auc_score(val_y_true, val_y_probs_auc)
            else:
                # Se for Multi-Classe real (> 2 classes), usamos a versão OVR com todas as colunas
                auc_val_geral = roc_auc_score(val_y_true, val_y_probs, average='weighted', multi_class='ovr')

            fold_metrics['auc_validation_geral'] = float(auc_val_geral)
            save_roc_curve_vectors(val_y_true, val_y_probs, 'Validação', fold_dir)
            
            print(f"  -> Limiar Ótimo (Youden's J) encontrado: {optimal_threshold_for_fold:.4f}")
            print(f"  -> AUC Validação Geral: {auc_val_geral:.4f}")

        else:
            # Caso raro de apenas uma classe no fold de validação
            fold_metrics['optimal_threshold'] = 0.5
            fold_metrics['auc_validation_geral'] = np.nan
            print("  -> AVISO: Não foi possível calcular AUC/limiar (apenas uma classe na validação). Usando 0.5.")
            
        # 4. Cálculo de Métricas de Classificação (usando o Limiar Ótimo)
        threshold = fold_metrics['optimal_threshold']
        
        # Para Multi-Classe, a predição é o argmax (maior probabilidade), não um limiar binário
        y_pred_class = np.argmax(val_y_probs, axis=1) # Predição Multiclasse
        
        # Para a confusão, precisamos tratar como binário para Sensibilidade/Especificidade, ou focar em Accuracy/F1-score.
        # Mantendo o Sens/Spec do código original (assumindo 2 classes)
        # Se for Multi-Classe real, você DEVE trocar por Accuracy, F1-Score e matriz de confusão Multi-Classe.
        if num_classes == 2:
             tn, fp, fn, tp = confusion_matrix(val_y_true, y_pred_class, labels=[0, 1]).ravel()
             sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
             specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
             fold_metrics[f'sensitivity_validation'] = float(sensitivity)
             fold_metrics[f'specificity_validation'] = float(specificity)
        
        # O resultado do Fold agora inclui o limiar e o desempenho de validação
        fold_results.append(fold_metrics)
        
    # --- FIM DO LOOP DE K-FOLD ---
    
    # --- 5. Seleção do Melhor Modelo e Avaliação Hold-Out ---
    
    # Seleção do melhor modelo (Baseado na Média AUC de validação se não houver um 'Operação' definido)
    # Aqui vamos usar a média do AUC de validação como critério de seleção simples
    best_fold_index = np.nanargmax([res.get('auc_validation_geral', 0.0) for res in fold_results])
    best_fold_num = best_fold_index + 1
    best_model_threshold = fold_results[best_fold_index].get('optimal_threshold', 0.5)

    print("\n" + "="*60)
    print("AVALIAÇÃO FINAL DO MELHOR MODELO NO CONJUNTO HOLD-OUT")
    
    # Carrega o modelo campeão
    best_model_path = os.path.join(experiment_dir, f"fold_{best_fold_num}", 'best_model.pt')
    final_model_save_path = os.path.join(experiment_dir, 'best_overall_model.pt')
    shutil.copy(best_model_path, final_model_save_path)
    
    final_model = model_class(input_shape=X_data.shape[1] if X_data.ndim == 2 else X_data.shape[1:], 
                              num_classes=num_classes, 
                              config=config).to(device)
    final_model.load_state_dict(torch.load(best_model_path))
    final_model.eval()

    # --- INÍCIO DA CORREÇÃO DE LEAKAGE (HOLD-OUT) ---
    # Carregar o scaler do melhor fold para transformar o Holdout (X_test_raw)
    import joblib # Necessário para carregar o scaler
    best_scaler_path = os.path.join(experiment_dir, f"fold_{best_fold_num}", 'fold_scaler.pkl')
    final_scaler = joblib.load(best_scaler_path)
    
    # Aplicar a transformação no conjunto Holdout (RAW)
    #X_holdout_scaled = final_scaler.transform(X_test_raw)
    X_holdout_scaled = _scale_data_safely(X_test_raw, final_scaler, fit=False)
    # Criar o DataLoader do Holdout (Teste Final) com dados ESCALONADOS
    holdout_dataset = dataset_class(X_holdout_scaled, Y_test)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    # --- FIM DA CORREÇÃO DE LEAKAGE (HOLD-OUT) ---
    
    # Executa a avaliação no hold-out
    holdout_y_true, holdout_probs = [], []
    with torch.no_grad():
        iterator = tqdm(holdout_loader, desc="  Avaliação Final no Hold-Out", unit="batch")
        for X_batch, Y_batch in iterator:
            X_batch = X_batch.to(device)
            outputs = final_model(X_batch)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            holdout_probs.extend(probabilities)
            holdout_y_true.extend(Y_batch.cpu().numpy())

    holdout_y_true = np.array(holdout_y_true)
    holdout_probs = np.array(holdout_probs)
    
    final_holdout_metrics = {}
    holdout_dir = os.path.join(experiment_dir, "holdout_results")
    os.makedirs(holdout_dir, exist_ok=True)
    
    # Cálculo das Métricas Hold-Out
    if len(np.unique(holdout_y_true)) > 1:
        #auc_holdout = roc_auc_score(holdout_y_true, holdout_probs, average='weighted', multi_class='ovr')
        if num_classes == 2:
            auc_holdout = roc_auc_score(holdout_y_true, holdout_probs[:, 1]) # Vers�o Bin�ria
        else:
            auc_holdout = roc_auc_score(holdout_y_true, holdout_probs, average='weighted', multi_class='ovr') # Vers�o Multiclasse
            final_holdout_metrics[f'auc_holdout_geral'] = auc_holdout
        
        plot_roc_curve(holdout_y_true, holdout_probs, "Final", "Hold-Out", holdout_dir)
        save_roc_curve_vectors(holdout_y_true, holdout_probs, "Hold-Out", holdout_dir)
        
        # Métricas de classificação usando o Limiar Ótimo do MELHOR FOLD
        y_pred_class = np.argmax(holdout_probs, axis=1) # Predição Multiclasse
        
        # Se for Multi-Classe, Sensibilidade/Especificidade não é a métrica padrão.
        # Mantemos o cálculo apenas para o caso binário (num_classes == 2)
        if num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(holdout_y_true, y_pred_class, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            final_holdout_metrics[f'sensitivity_holdout'] = float(sensitivity)
            final_holdout_metrics[f'specificity_holdout'] = float(specificity)
        
        print(f"  -> AUC Hold-Out Geral: {auc_holdout:.4f}")
        
    else:
        print("  -> AVISO: Não foi possível calcular AUC no Hold-Out (apenas uma classe).")

    # --- Sumarização Final ---
    cv_summary_dict = {}
    metrics_to_summarize = [key for key in fold_results[0].keys()] 
    
    for key in metrics_to_summarize:
        values = [res.get(key, np.nan) for res in fold_results]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        cv_summary_dict[f'mean_{key}'] = float(mean_val)
        cv_summary_dict[f'std_{key}'] = float(std_val)

    final_results = {
        'cross_validation_summary': cv_summary_dict,
        'final_holdout_results': final_holdout_metrics
    }
    
    # Salvar resultados finais como JSON
    with open(os.path.join(experiment_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Resultados finais salvos em: {os.path.join(experiment_dir, 'final_metrics.json')}")

    print("\n" + "="*60)
    print(f"SUMÁRIO FINAL PARA O MODELO: {model_name}")
    print(f"AUC Validação Geral (Média CV): {cv_summary_dict['mean_auc_validation_geral']:.4f} (+/- {cv_summary_dict['std_auc_validation_geral']:.4f})")
    #print(f"AUC Hold-Out Geral (Final): {final_holdout_metrics.get('auc_holdout_geral', 'N/A'):.4f}")
    # CORRE��O: Trate o valor padr�o 'N/A' como uma string separada
    auc_holdout_value = final_holdout_metrics.get('auc_holdout_geral', 'N/A')

    if auc_holdout_value == 'N/A':
        formatted_auc = 'N/A'
    else:
        # Formate apenas se for um n�mero (float)
        formatted_auc = f"{auc_holdout_value:.4f}"

    print(f"AUC Hold-Out Geral (Final): {formatted_auc}")
    print("="*60)
    
    return final_results