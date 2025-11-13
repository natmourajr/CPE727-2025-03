import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configura��o para encontrar a pasta 'data/processed/'
# Ajuste o n�mero de .parent() conforme necess�rio para a sua estrutura
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
EXAMPLES_DIR = PROJECT_ROOT / 'results' / 'examples'
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# CONFIGURA��O DO MATPLOTLIB (ESSENCIAL PARA CONTAINERS)
# ------------------------------------------------
try:
    plt.switch_backend('Agg') # Usa backend que n�o requer interface gr�fica
except ImportError:
    print("Aviso: Matplotlib n�o encontrado ou com erro de importa��o.")
    
def generate_cnn_image_example(n_examples=4):
    """Carrega o array 2D e gera uma imagem de exemplo."""
    
    try:
        # Carregar os dados processados
        X_images = np.load(PROCESSED_DATA_DIR / 'X_image.npy')
        # Carregar os r�tulos originais
        Y_encoded = np.load(PROCESSED_DATA_DIR / 'Y_final.npy')
    except FileNotFoundError:
        print("ERRO: Arquivos .npy n�o encontrados. Execute data_processor.py primeiro.")
        return

    # Usaremos os primeiros N exemplos
    images_to_plot = X_images[:n_examples]
    labels_to_plot = Y_encoded[:n_examples]
    
    side = images_to_plot.shape[1]

    plt.figure(figsize=(n_examples * 3, 3))

    for i in range(n_examples):
        ax = plt.subplot(1, n_examples, i + 1)
        
        # Plota a matriz 2D (a "imagem")
        plt.imshow(images_to_plot[i], cmap='gray', interpolation='nearest')
        
        ax.set_title(f"Amostra {i+1}", fontsize=10)
        ax.set_xticks(np.arange(side))
        ax.set_yticks(np.arange(side))
        
        # Remove labels/ticks para uma visualiza��o limpa
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    output_path = EXAMPLES_DIR / 'image_transformation_examples.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\n\u2705 Imagens de exemplo para o relat�rio salvas em: {output_path}")

if __name__ == '__main__':
    generate_cnn_image_example(n_examples=4)