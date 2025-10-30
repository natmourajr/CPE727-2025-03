"""
Demonstração de Overfitting vs Dropout Regularization
Extraído do notebook overfitting_mnist.ipynb

Este script demonstra o efeito do overfitting em redes neurais profundas 
e como a técnica de Dropout pode mitigar esse problema.

Configuração:
- Dataset: MNIST (subset de 1000 amostras de treino)
- Arquitetura: 9 camadas densas (2048→1024→1024→512→512→256→256→128→64→10)
- Dropout rate: 0.3
- Épocas: 100
- Batch size: 32
"""

# ============================================================================
# 1. Importação de Bibliotecas
# ============================================================================
import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input,BatchNormalization, Activation
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")

# ============================================================================
# 2. Carregamento e Preparação dos Dados
# ============================================================================

# Carregar MNIST
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Normalizar para [0, 1]
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten: 28x28 -> 784
x_train_full = x_train_full.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Criar subset pequeno para facilitar overfitting
subset_size = 1000
x_train_subset = x_train_full[:subset_size]
y_train_subset = y_train_full[:subset_size]

print(f"Training subset: {x_train_subset.shape}, {y_train_subset.shape}")
print(f"Test set: {x_test.shape}, {y_test.shape}")

# ============================================================================
# 3. Definição dos Modelos
# ============================================================================

def create_model_without_dropout():
    """Modelo sem Dropout - propenso a overfitting"""
    model = Sequential([
        Input(shape=(784,)),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_model_with_dropout():
    """Modelo com Dropout - regularização para melhor generalização"""
    model = Sequential([
        Dense(2048, input_shape=(784,)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Modelos definidos com sucesso!")

# ============================================================================
# 4. Treinamento dos Modelos
# ============================================================================


# Hiperparâmetros
epochs = 50
batch_size = 32

print("=" * 60)
print("EXPERIMENTO 1: Modelo SEM Dropout")
print("=" * 60)

model_no_dropout = create_model_without_dropout()
history_no_dropout = model_no_dropout.fit(
    x_train_subset, y_train_subset,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\n" + "=" * 60)
print("EXPERIMENTO 2: Modelo COM Dropout")
print("=" * 60)

model_with_dropout = create_model_with_dropout()
history_with_dropout = model_with_dropout.fit(
    x_train_subset, y_train_subset,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\nTreinamento concluído!")

# ============================================================================
# 5. Visualização dos Resultados
# ============================================================================

# Criar pasta de resultados com timestamp
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', timestamp_str)
os.makedirs(results_dir, exist_ok=True)

print(f"\n📁 Criando pasta de resultados: {results_dir}")

# Extrair métricas finais
train_acc_no_dropout = history_no_dropout.history['accuracy'][-1]
test_acc_no_dropout = history_no_dropout.history['val_accuracy'][-1]
train_loss_no_dropout = history_no_dropout.history['loss'][-1]
test_loss_no_dropout = history_no_dropout.history['val_loss'][-1]

train_acc_with_dropout = history_with_dropout.history['accuracy'][-1]
test_acc_with_dropout = history_with_dropout.history['val_accuracy'][-1]
train_loss_with_dropout = history_with_dropout.history['loss'][-1]
test_loss_with_dropout = history_with_dropout.history['val_loss'][-1]

# Criar visualizações
fig = plt.figure(figsize=(20, 12))

# 1. Loss sem dropout
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history_no_dropout.history['loss'], label='Treino', linewidth=2)
ax1.plot(history_no_dropout.history['val_loss'], label='Teste', linewidth=2)
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Experimento 1: Loss (Sem Dropout)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Accuracy sem dropout
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history_no_dropout.history['accuracy'], label='Treino', linewidth=2)
ax2.plot(history_no_dropout.history['val_accuracy'], label='Teste', linewidth=2)
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Acurácia', fontsize=12)
ax2.set_title('Experimento 1: Acurácia (Sem Dropout)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Loss com dropout
ax3 = plt.subplot(2, 3, 4)
ax3.plot(history_with_dropout.history['loss'], label='Treino', linewidth=2)
ax3.plot(history_with_dropout.history['val_loss'], label='Teste', linewidth=2)
ax3.set_xlabel('Época', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Experimento 2: Loss (Com Dropout)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Accuracy com dropout
ax4 = plt.subplot(2, 3, 5)
ax4.plot(history_with_dropout.history['accuracy'], label='Treino', linewidth=2)
ax4.plot(history_with_dropout.history['val_accuracy'], label='Teste', linewidth=2)
ax4.set_xlabel('Época', fontsize=12)
ax4.set_ylabel('Acurácia', fontsize=12)
ax4.set_title('Experimento 2: Acurácia (Com Dropout)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# 5. Comparação final
ax5 = plt.subplot(1, 3, 3)
x_pos = np.arange(4)
width = 0.35

bars_no_dropout = [train_acc_no_dropout, test_acc_no_dropout, 
                   train_loss_no_dropout, test_loss_no_dropout]
bars_with_dropout = [train_acc_with_dropout, test_acc_with_dropout, 
                     train_loss_with_dropout, test_loss_with_dropout]

bar1 = ax5.bar(x_pos - width/2, bars_no_dropout, width, 
               label='Sem Dropout', alpha=0.8, color='#e74c3c')
bar2 = ax5.bar(x_pos + width/2, bars_with_dropout, width, 
               label='Com Dropout', alpha=0.8, color='#3498db')

ax5.set_xlabel('Métrica', fontsize=12)
ax5.set_ylabel('Valor', fontsize=12)
ax5.set_title('Comparação Final das Métricas', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['Acc Treino', 'Acc Teste', 'Loss Treino', 'Loss Teste'], 
                     rotation=15, ha='right')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plot_filename = os.path.join(results_dir, "overfitting_comparison.png")
fig.savefig(plot_filename, dpi=300, bbox_inches='tight')  # ✅ Salva UMA vez
print(f"✅ Gráfico salvo em: {plot_filename}")

plt.show()

print("\n" + "=" * 60)
print("RESULTADOS FINAIS")
print("=" * 60)
print(f"\nSEM DROPOUT:")
print(f"  Acurácia Treino: {train_acc_no_dropout:.3f}")
print(f"  Acurácia Teste:  {test_acc_no_dropout:.3f}")
print(f"  Gap (Overfitting): {train_acc_no_dropout - test_acc_no_dropout:.3f}")
print(f"  Loss Treino: {train_loss_no_dropout:.3f}")
print(f"  Loss Teste:  {test_loss_no_dropout:.3f}")

print(f"\nCOM DROPOUT:")
print(f"  Acurácia Treino: {train_acc_with_dropout:.3f}")
print(f"  Acurácia Teste:  {test_acc_with_dropout:.3f}")
print(f"  Gap (Overfitting): {train_acc_with_dropout - test_acc_with_dropout:.3f}")
print(f"  Loss Treino: {train_loss_with_dropout:.3f}")
print(f"  Loss Teste:  {test_loss_with_dropout:.3f}")

print("\n" + "=" * 60)
print("ANÁLISE")
print("=" * 60)
print(f"Redução no Gap de Acurácia: "
      f"{((train_acc_no_dropout - test_acc_no_dropout) - (train_acc_with_dropout - test_acc_with_dropout)):.3f}")
print(f"Melhoria na Acurácia de Teste: "
      f"{(test_acc_with_dropout - test_acc_no_dropout):.3f}")
print("=" * 60)

# ============================================================================
# 6. Salvar Resultados em Log
# ============================================================================



# Preparar dados para salvar
experiment_log = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "configuration": {
        "subset_size": subset_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate": 0.3,
        "optimizer": "SGD",
        "learning_rate": 0.01
    },
    "sem_dropout": {
        "train_accuracy": float(train_acc_no_dropout),
        "test_accuracy": float(test_acc_no_dropout),
        "train_loss": float(train_loss_no_dropout),
        "test_loss": float(test_loss_no_dropout),
        "gap_accuracy": float(train_acc_no_dropout - test_acc_no_dropout)
    },
    "com_dropout": {
        "train_accuracy": float(train_acc_with_dropout),
        "test_accuracy": float(test_acc_with_dropout),
        "train_loss": float(train_loss_with_dropout),
        "test_loss": float(test_loss_with_dropout),
        "gap_accuracy": float(train_acc_with_dropout - test_acc_with_dropout)
    },
    "analise": {
        "reducao_gap_accuracy": float((train_acc_no_dropout - test_acc_no_dropout) - 
                                       (train_acc_with_dropout - test_acc_with_dropout)),
        "melhoria_test_accuracy": float(test_acc_with_dropout - test_acc_no_dropout)
    }
}

# Salvar em arquivo JSON
log_filename = os.path.join(results_dir, "experiment_log.json")
with open(log_filename, 'w', encoding='utf-8') as f:
    json.dump(experiment_log, f, indent=4, ensure_ascii=False)

print(f"✅ Resultados salvos em: {log_filename}")

# Também salvar histórico completo (todas as épocas)
history_log = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "configuration": experiment_log["configuration"],
    "sem_dropout": {
        "train_accuracy_history": [float(x) for x in history_no_dropout.history['accuracy']],
        "test_accuracy_history": [float(x) for x in history_no_dropout.history['val_accuracy']],
        "train_loss_history": [float(x) for x in history_no_dropout.history['loss']],
        "test_loss_history": [float(x) for x in history_no_dropout.history['val_loss']]
    },
    "com_dropout": {
        "train_accuracy_history": [float(x) for x in history_with_dropout.history['accuracy']],
        "test_accuracy_history": [float(x) for x in history_with_dropout.history['val_accuracy']],
        "train_loss_history": [float(x) for x in history_with_dropout.history['loss']],
        "test_loss_history": [float(x) for x in history_with_dropout.history['val_loss']]
    }
}

history_filename = os.path.join(results_dir, "experiment_history.json")
with open(history_filename, 'w', encoding='utf-8') as f:
    json.dump(history_log, f, indent=4, ensure_ascii=False)

print(f"✅ Histórico completo salvo em: {history_filename}")

# Criar também um log de texto simples
txt_filename = os.path.join(results_dir, "experiment_summary.txt")
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("EXPERIMENTO: Overfitting vs Dropout Regularization\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("CONFIGURAÇÃO:\n")
    f.write(f"  - Subset size: {subset_size}\n")
    f.write(f"  - Épocas: {epochs}\n")
    f.write(f"  - Batch size: {batch_size}\n")
    f.write(f"  - Dropout rate: 0.3\n")
    f.write(f"  - Optimizer: SGD (lr=0.01)\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("RESULTADOS FINAIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("SEM DROPOUT:\n")
    f.write(f"  Acurácia Treino:     {train_acc_no_dropout:.3f}\n")
    f.write(f"  Acurácia Teste:      {test_acc_no_dropout:.3f}\n")
    f.write(f"  Gap (Overfitting):   {train_acc_no_dropout - test_acc_no_dropout:.3f}\n")
    f.write(f"  Loss Treino:         {train_loss_no_dropout:.3f}\n")
    f.write(f"  Loss Teste:          {test_loss_no_dropout:.3f}\n\n")
    
    f.write("COM DROPOUT:\n")
    f.write(f"  Acurácia Treino:     {train_acc_with_dropout:.3f}\n")
    f.write(f"  Acurácia Teste:      {test_acc_with_dropout:.3f}\n")
    f.write(f"  Gap (Overfitting):   {train_acc_with_dropout - test_acc_with_dropout:.3f}\n")
    f.write(f"  Loss Treino:         {train_loss_with_dropout:.3f}\n")
    f.write(f"  Loss Teste:          {test_loss_with_dropout:.3f}\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("ANÁLISE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Redução no Gap de Acurácia:  "
            f"{((train_acc_no_dropout - test_acc_no_dropout) - (train_acc_with_dropout - test_acc_with_dropout)):.3f}\n")
    f.write(f"Melhoria na Acurácia de Teste: "
            f"{(test_acc_with_dropout - test_acc_no_dropout):.3f}\n\n")
    
    f.write("=" * 70 + "\n")

print(f"✅ Resumo em texto salvo em: {txt_filename}")

print("\n" + "=" * 60)
print("EXPERIMENTO CONCLUÍDO!")
print(f"📁 Todos os resultados salvos em: {results_dir}")
print("=" * 60)
