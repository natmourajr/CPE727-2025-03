"""
Script para Visualizar Feature Maps do SimpleCNN
Mostra o que cada camada convolucional detecta em imagens de TB
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Importar modelo
# Adicionar diretório raiz do projeto ao path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SimpleCNN

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path('models/simplecnn_best.pth')
DATA_DIR = Path('data/shenzhen')
OUTPUT_DIR = Path('results/feature_maps')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Transformações (mesmas do treinamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


def load_model():
    """Carrega modelo SimpleCNN treinado"""
    print("Carregando modelo SimpleCNN...")
    model = SimpleCNN(num_classes=2, dropout=0.4)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modelo carregado de {MODEL_PATH}")
    else:
        print(f"⚠️ Modelo não encontrado em {MODEL_PATH}")
        return None
    
    model = model.to(DEVICE)
    model.eval()
    return model


def get_feature_maps(model, image_tensor):
    """
    Extrai feature maps de cada bloco convolucional
    Retorna dicionário com feature maps de cada camada
    """
    feature_maps = {}
    x = image_tensor.to(DEVICE)
    
    # Bloco 1 - Sequential: Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool2d
    # Extrair features após o segundo ReLU (antes do pooling)
    for i, layer in enumerate(model.conv1):
        x = layer(x)
        if i == 5:  # Após o segundo ReLU
            feature_maps['block1'] = x.detach().cpu()
    
    # Bloco 2
    for i, layer in enumerate(model.conv2):
        x = layer(x)
        if i == 5:  # Após o segundo ReLU
            feature_maps['block2'] = x.detach().cpu()
    
    # Bloco 3
    for i, layer in enumerate(model.conv3):
        x = layer(x)
        if i == 5:  # Após o segundo ReLU
            feature_maps['block3'] = x.detach().cpu()
    
    # Bloco 4
    for i, layer in enumerate(model.conv4):
        x = layer(x)
        if i == 5:  # Após o segundo ReLU
            feature_maps['block4'] = x.detach().cpu()
    
    return feature_maps


def visualize_feature_maps_grid(feature_maps, image_name, save_path):
    """
    Visualiza feature maps de todos os blocos em uma grade
    Mostra os primeiros 16 filtros de cada bloco
    """
    blocks = ['block1', 'block2', 'block3', 'block4']
    num_filters_to_show = 16
    
    fig, axes = plt.subplots(4, num_filters_to_show, figsize=(20, 8))
    fig.suptitle(f'Feature Maps - {image_name}', fontsize=16, fontweight='bold')
    
    for block_idx, block_name in enumerate(blocks):
        fmaps = feature_maps[block_name][0]  # Remove batch dimension
        num_channels = fmaps.shape[0]
        
        for filter_idx in range(num_filters_to_show):
            ax = axes[block_idx, filter_idx]
            
            if filter_idx < num_channels:
                # Plotar feature map
                fmap = fmaps[filter_idx].numpy()
                im = ax.imshow(fmap, cmap='viridis')
                ax.axis('off')
                
                # Título apenas na primeira coluna
                if filter_idx == 0:
                    ax.set_ylabel(f'{block_name}\n({num_channels} filtros)', 
                                fontweight='bold', fontsize=10)
            else:
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature maps salvos: {save_path}")
    plt.close()


def visualize_feature_maps_detailed(feature_maps, block_name, image_name, save_path):
    """
    Visualiza todos os filtros de um bloco específico
    """
    fmaps = feature_maps[block_name][0]  # Remove batch dimension
    num_channels = fmaps.shape[0]
    
    # Calcular grid size
    cols = 8
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))
    fig.suptitle(f'{block_name} - {image_name} ({num_channels} filtros)', 
                fontsize=14, fontweight='bold')
    
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        if idx < num_channels:
            fmap = fmaps[idx].numpy()
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Filtro {idx+1}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ {block_name} detalhado salvo: {save_path}")
    plt.close()


def visualize_activation_progression(original_image, feature_maps, image_name, save_path):
    """
    Mostra progressão de ativações através das camadas
    Exibe imagem original + média dos feature maps de cada bloco
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'Progressão de Ativações - {image_name}', 
                fontsize=14, fontweight='bold')
    
    # Imagem original
    axes[0].imshow(original_image)
    axes[0].set_title('Imagem Original', fontweight='bold')
    axes[0].axis('off')
    
    # Feature maps médios de cada bloco
    blocks = ['bloco1', 'bloco2', 'bloco3', 'bloco4']
    for idx, block_name in enumerate(blocks, start=1):
        fmaps = feature_maps[block_name][0]  # Remove batch dimension
        
        # Calcular média dos feature maps
        mean_fmap = torch.mean(fmaps, dim=0).numpy()
        
        axes[idx].imshow(mean_fmap, cmap='viridis')
        axes[idx].set_title(f'{block_name}\n({fmaps.shape[0]} filtros, {fmaps.shape[1]}x{fmaps.shape[2]})', 
                          fontweight='bold', fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Progressão de ativações salva: {save_path}")
    plt.close()


def compare_tb_vs_normal(model, tb_image_path, normal_image_path):
    """
    Compara feature maps entre imagem com TB e normal
    """
    print("\n" + "="*80)
    print("COMPARAÇÃO: TB vs Normal")
    print("="*80)
    
    # Carregar imagens
    tb_image = Image.open(tb_image_path).convert('RGB')
    normal_image = Image.open(normal_image_path).convert('RGB')
    
    # Transformar
    tb_tensor = transform(tb_image).unsqueeze(0)
    normal_tensor = transform(normal_image).unsqueeze(0)
    
    # Extrair feature maps
    print("Extraindo feature maps da imagem com TB...")
    tb_fmaps = get_feature_maps(model, tb_tensor)
    
    print("Extraindo feature maps da imagem normal...")
    normal_fmaps = get_feature_maps(model, normal_tensor)
    
    # Visualizar comparação lado a lado
    blocks = ['block1', 'block2', 'block3', 'block4']
    num_filters_to_show = 8
    
    # Criar grid com espaço no meio: 8 (TB) + 1 (Espaço) + 8 (Normal) = 17 colunas
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(len(blocks), 17)
    
    # Adicionar Títulos de Seção
    plt.figtext(0.28, 0.92, "PACIENTE COM TB (Filtros 1-8)", ha='center', fontsize=16, fontweight='bold', color='darkred')
    plt.figtext(0.72, 0.92, "PACIENTE NORMAL (Filtros 1-8)", ha='center', fontsize=16, fontweight='bold', color='darkgreen')
    
    # Adicionar linha vertical divisória no meio exato
    line = plt.Line2D([0.5, 0.5], [0.1, 0.9], transform=fig.transFigure, color="black", linewidth=2, linestyle='--')
    fig.add_artist(line)

    fig.suptitle('Comparação de Ativações: Como a rede vê cada caso', 
                fontsize=20, fontweight='bold', y=0.98)
    
    for block_idx, block_name in enumerate(blocks):
        tb_fmap = tb_fmaps[block_name][0]
        normal_fmap = normal_fmaps[block_name][0]
        
        # Plotar TB (Colunas 0-7)
        for filter_idx in range(num_filters_to_show):
            ax = fig.add_subplot(gs[block_idx, filter_idx])
            
            if filter_idx < tb_fmap.shape[0]:
                ax.imshow(tb_fmap[filter_idx].numpy(), cmap='viridis')
                ax.axis('off')
                
                # Labels de linha (Blocos)
                if filter_idx == 0:
                    ax.set_ylabel(f'{block_name}', fontweight='bold', fontsize=12)
                    ax.text(-0.5, 0.5, f'{block_name}', va='center', ha='right', 
                              transform=ax.transAxes, fontweight='bold', fontsize=11, rotation=90)
                
                # Títulos (apenas primeira linha)
                if block_idx == 0:
                     ax.set_title(f'Filtro {filter_idx+1}', fontsize=10)
        
        # Plotar Normal (Colunas 9-16) -> Pula a coluna 8
        for filter_idx in range(num_filters_to_show):
            ax = fig.add_subplot(gs[block_idx, filter_idx + 9]) # +9 pulando o meio
            
            if filter_idx < normal_fmap.shape[0]:
                ax.imshow(normal_fmap[filter_idx].numpy(), cmap='viridis')
                ax.axis('off')
                
                if block_idx == 0:
                     ax.set_title(f'Filtro {filter_idx+1}', fontsize=10)
    
    plt.subplots_adjust(top=0.9, wspace=0.1, hspace=0.05)
    
    save_path = OUTPUT_DIR / 'comparison_tb_vs_normal.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparação salva: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("VISUALIZAÇÃO DE FEATURE MAPS - SimpleCNN")
    print("="*80)
    
    # Carregar modelo
    model = load_model()
    if model is None:
        print("❌ Não foi possível carregar o modelo. Verifique o caminho.")
        return
    
    # Encontrar imagens de exemplo
    tb_images = list((DATA_DIR / 'tuberculosis').glob('*.png'))
    normal_images = list((DATA_DIR / 'normal').glob('*.png'))
    
    if not tb_images or not normal_images:
        print(f"⚠️ Imagens não encontradas em {DATA_DIR}")
        print("Certifique-se de que o dataset está organizado em:")
        print("  data/shenzhen/tuberculosis/*.png")
        print("  data/shenzhen/normal/*.png")
        return
    
    # Selecionar primeira imagem de cada classe
    tb_image_path = tb_images[0]
    normal_image_path = normal_images[0]
    
    print(f"\nImagem TB: {tb_image_path.name}")
    print(f"Imagem Normal: {normal_image_path.name}")
    
    # Processar imagem com TB
    print("\n" + "="*80)
    print("PROCESSANDO IMAGEM COM TB")
    print("="*80)
    
    tb_image = Image.open(tb_image_path).convert('RGB')
    tb_tensor = transform(tb_image).unsqueeze(0)
    tb_fmaps = get_feature_maps(model, tb_tensor)
    
    # Visualizações para TB
    # visualize_feature_maps_grid(tb_fmaps, f"TB - {tb_image_path.name}", 
    #                            OUTPUT_DIR / 'tb_feature_maps_grid.png')
    
    # visualize_activation_progression(tb_image, tb_fmaps, f"TB - {tb_image_path.name}",
    #                                 OUTPUT_DIR / 'tb_activation_progression.png')
    
    # Detalhado de cada bloco (TB)
    # for block_name in ['block1', 'block2', 'block3', 'block4']:
    #     visualize_feature_maps_detailed(tb_fmaps, block_name, f"TB - {tb_image_path.name}",
    #                                    OUTPUT_DIR / f'tb_{block_name}_detailed.png')
    
    # Processar imagem Normal
    print("\n" + "="*80)
    print("PROCESSANDO IMAGEM NORMAL")
    print("="*80)
    
    normal_image = Image.open(normal_image_path).convert('RGB')
    normal_tensor = transform(normal_image).unsqueeze(0)
    normal_fmaps = get_feature_maps(model, normal_tensor)
    
    # Visualizações para Normal
    # visualize_feature_maps_grid(normal_fmaps, f"Normal - {normal_image_path.name}",
    #                            OUTPUT_DIR / 'normal_feature_maps_grid.png')
    
    # visualize_activation_progression(normal_image, normal_fmaps, f"Normal - {normal_image_path.name}",
    #                                 OUTPUT_DIR / 'normal_activation_progression.png')
    
    # Comparação TB vs Normal
    compare_tb_vs_normal(model, tb_image_path, normal_image_path)
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO")
    print("="*80)
    print(f"\n✅ Todas as visualizações foram geradas em: {OUTPUT_DIR}")
    print("\nArquivos criados:")
    print("  📊 Grids de Feature Maps:")
    print("     • tb_feature_maps_grid.png")
    print("     • normal_feature_maps_grid.png")
    print("\n  📈 Progressão de Ativações:")
    print("     • tb_activation_progression.png")
    print("     • normal_activation_progression.png")
    print("\n  🔍 Detalhados por Bloco (TB):")
    print("     • tb_block1_detailed.png (32 filtros)")
    print("     • tb_block2_detailed.png (64 filtros)")
    print("     • tb_block3_detailed.png (128 filtros)")
    print("     • tb_block4_detailed.png (256 filtros)")
    print("\n  ⚖️ Comparação:")
    print("     • comparison_tb_vs_normal.png")
    
    print("\n" + "="*80)
    print("✅ Visualização concluída!")
    print("="*80)


if __name__ == '__main__':
    main()
