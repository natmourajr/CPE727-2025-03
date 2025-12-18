import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' 

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_class_pie_chart():
    """
    Carrega os rótulos, calcula a distribuição e plota o gráfico de pizza.
    """

    Y_PATH = PROCESSED_DATA_DIR / 'Y_final.npy'
    LE_PATH = PROCESSED_DATA_DIR / 'label_encoder.pkl'
    
    if not Y_PATH.exists():
        print(f"ERRO: Arquivo de rótulos não encontrado em {Y_PATH}")
        return

    Y_data = np.load(Y_PATH)
    
    try:
       
        le = joblib.load(LE_PATH)
        class_labels = le.classes_
    except FileNotFoundError:
        print("AVISO: LabelEncoder não encontrado. Usando rótulos numéricos.")
        
        class_labels = [f"Classe {i}" for i in np.unique(Y_data)]
    except Exception as e:
        print(f"Erro ao carregar LabelEncoder: {e}. Usando rótulos numéricos.")
        class_labels = [f"Classe {i}" for i in np.unique(Y_data)]

   
    unique, counts = np.unique(Y_data, return_counts=True)
    distribution = dict(zip(unique, counts))
    
   
    labels = [f"{class_labels[i]} ({count})" for i, count in distribution.items()]
    sizes = list(distribution.values())
    
    
    colors = plt.cm.Set3.colors 

   
    plt.figure(figsize=(10, 8))
    
   
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors[:len(sizes)]
    )
    
    plt.title('Distribuição de Classes no Dataset CIC-MalMem-2022', fontsize=16)
    
   
    plt.axis('equal') 
    
   
    save_path = RESULTS_DIR / 'class_distribution_pie_chart.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print(f"✅ Distribuição de Classes Plotada com Sucesso!")
    print(f"Gráfico salvo em: {save_path}")
    print("="*50)
    
    print("\nDistribuição de Amostras:")
    for label, count in zip(class_labels, counts):
        print(f"  {label}: {count} amostras")
    

if __name__ == '__main__':
    plot_class_pie_chart()