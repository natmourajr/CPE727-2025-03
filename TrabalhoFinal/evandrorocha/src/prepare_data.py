"""
Script para preparar e organizar o dataset Shenzhen
"""
import os
import shutil
from pathlib import Path
import pandas as pd


def organize_shenzhen_dataset(source_dir: str, target_dir: str):
    """
    Organiza o dataset Shenzhen na estrutura necessária
    
    Args:
        source_dir: Diretório contendo as imagens originais
        target_dir: Diretório de destino
    """
    
    # Criar estrutura de diretórios
    normal_dir = os.path.join(target_dir, 'normal')
    tb_dir = os.path.join(target_dir, 'tuberculosis')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    # O dataset Shenzhen geralmente vem com um arquivo de metadados
    # indicando quais imagens são normais e quais têm TB
    # Você precisará adaptar isso baseado na estrutura real do download
    
    print("Organizando dataset...")
    
    # Exemplo de organização baseada em nome de arquivo
    # Os arquivos geralmente têm padrão: CHNCXR_XXXX_0.png (normal) ou CHNCXR_XXXX_1.png (TB)
    
    source_path = Path(source_dir)
    
    for img_file in source_path.glob('*.png'):
        filename = img_file.name
        
        # Verificar se é normal (0) ou TB (1) baseado no padrão do nome
        if '_0.' in filename:
            # Normal
            shutil.copy(img_file, os.path.join(normal_dir, filename))
        elif '_1.' in filename:
            # Tuberculosis
            shutil.copy(img_file, os.path.join(tb_dir, filename))
    
    # Contar arquivos
    normal_count = len(list(Path(normal_dir).glob('*.png')))
    tb_count = len(list(Path(tb_dir).glob('*.png')))
    
    print(f"\nDataset organizado com sucesso!")
    print(f"Imagens normais: {normal_count}")
    print(f"Imagens com TB: {tb_count}")
    print(f"Total: {normal_count + tb_count}")


def verify_dataset(data_dir: str):
    """Verifica a integridade do dataset"""
    normal_dir = os.path.join(data_dir, 'normal')
    tb_dir = os.path.join(data_dir, 'tuberculosis')
    
    if not os.path.exists(normal_dir):
        print(f"❌ Diretório não encontrado: {normal_dir}")
        return False
    
    if not os.path.exists(tb_dir):
        print(f"❌ Diretório não encontrado: {tb_dir}")
        return False
    
    normal_count = len(list(Path(normal_dir).glob('*.png')))
    tb_count = len(list(Path(tb_dir).glob('*.png')))
    
    if normal_count == 0:
        print("❌ Nenhuma imagem normal encontrada")
        return False
    
    if tb_count == 0:
        print("❌ Nenhuma imagem com TB encontrada")
        return False
    
    print("✅ Dataset verificado com sucesso!")
    print(f"   - Imagens normais: {normal_count}")
    print(f"   - Imagens com TB: {tb_count}")
    print(f"   - Total: {normal_count + tb_count}")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar dataset Shenzhen')
    parser.add_argument('--source', type=str, required=True, 
                       help='Diretório com imagens originais')
    parser.add_argument('--target', type=str, default='./data/shenzhen',
                       help='Diretório de destino')
    parser.add_argument('--verify-only', action='store_true',
                       help='Apenas verificar dataset existente')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.target)
    else:
        organize_shenzhen_dataset(args.source, args.target)
        verify_dataset(args.target)
