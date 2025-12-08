"""
Script para organizar o dataset Shenzhen corretamente
Separa imagens em normal/ e tuberculosis/ baseado no nome do arquivo
"""
import os
import shutil
from pathlib import Path

def organize_shenzhen_dataset():
    """Organiza o dataset Shenzhen em pastas normal e tuberculosis"""
    
    # Diretórios
    source_dir = '/workspace/data/ChinaSet_AllFiles/CXR_png'
    dest_base = '/workspace/data/shenzhen'
    normal_dir = os.path.join(dest_base, 'normal')
    tb_dir = os.path.join(dest_base, 'tuberculosis')
    
    # Criar diretórios se não existirem
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    print('='*60)
    print('ORGANIZANDO DATASET SHENZHEN')
    print('='*60)
    print(f'\nOrigem: {source_dir}')
    print(f'Destino: {dest_base}')
    
    # Verificar se diretório fonte existe
    if not os.path.exists(source_dir):
        print(f'\n❌ Erro: Diretório fonte não encontrado: {source_dir}')
        return
    
    # Listar todas as imagens
    all_images = [f for f in os.listdir(source_dir) 
                  if f.endswith('.png')]
    
    print(f'\nTotal de imagens encontradas: {len(all_images)}')
    
    # Contadores
    normal_count = 0
    tb_count = 0
    
    # Organizar imagens
    for img_name in all_images:
        source_path = os.path.join(source_dir, img_name)
        
        # Determinar destino baseado no nome
        # Padrão: CHNCXR_XXXX_Y.png onde Y=0 (normal) ou Y=1 (TB)
        if img_name.endswith('_0.png'):
            dest_path = os.path.join(normal_dir, img_name)
            normal_count += 1
        elif img_name.endswith('_1.png'):
            dest_path = os.path.join(tb_dir, img_name)
            tb_count += 1
        else:
            print(f'⚠️  Ignorando arquivo com padrão desconhecido: {img_name}')
            continue
        
        # Copiar arquivo
        shutil.copy2(source_path, dest_path)
    
    print('\n' + '='*60)
    print('ORGANIZAÇÃO CONCLUÍDA!')
    print('='*60)
    print(f'\n✅ Imagens normais: {normal_count}')
    print(f'   Destino: {normal_dir}')
    print(f'\n✅ Imagens com TB: {tb_count}')
    print(f'   Destino: {tb_dir}')
    print(f'\n📊 Total: {normal_count + tb_count} imagens')
    
    # Verificar distribuição
    if normal_count > 0 and tb_count > 0:
        ratio = tb_count / normal_count
        print(f'\n📈 Proporção TB/Normal: {ratio:.2f}')
        
        if abs(ratio - 1.0) < 0.1:
            print('✨ Dataset balanceado!')
        else:
            print(f'⚠️  Dataset desbalanceado')
    
    print('\n' + '='*60)
    print('Dataset pronto para treinamento!')
    print('='*60)

if __name__ == '__main__':
    organize_shenzhen_dataset()
