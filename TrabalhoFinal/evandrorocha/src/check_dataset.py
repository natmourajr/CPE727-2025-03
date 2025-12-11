"""
Script para verificar e identificar imagens corrompidas no dataset
"""
import os
from PIL import Image
from pathlib import Path
import json

def check_dataset(data_dir='./data/shenzhen'):
    """Verifica todas as imagens do dataset e identifica as corrompidas"""
    
    corrupted_images = []
    valid_images = []
    
    # Diretórios a verificar
    dirs_to_check = {
        'normal': os.path.join(data_dir, 'normal'),
        'tuberculosis': os.path.join(data_dir, 'tuberculosis')
    }
    
    print('='*60)
    print('VERIFICAÇÃO DO DATASET SHENZHEN')
    print('='*60)
    
    for label, dir_path in dirs_to_check.items():
        if not os.path.exists(dir_path):
            print(f'\n⚠️  Diretório não encontrado: {dir_path}')
            continue
            
        print(f'\n📁 Verificando: {label}/')
        print('-'*60)
        
        image_files = [f for f in os.listdir(dir_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            img_path = os.path.join(dir_path, img_name)
            
            try:
                # Tentar abrir e verificar a imagem
                with Image.open(img_path) as img:
                    img.verify()  # Verifica integridade (deve ser chamado logo após open)
                
                # Reabrir para carregar completamente (verify fecha o arquivo)
                with Image.open(img_path) as img:
                    img.load()  # Força o carregamento completo
                    # Converter para RGB para garantir compatibilidade
                    img.convert('RGB')
                
                # Se chegou aqui, a imagem é válida
                valid_images.append({
                    'path': img_path,
                    'label': label,
                    'name': img_name
                })
                
            except Exception as e:
                # Imagem corrompida
                corrupted_images.append({
                    'path': img_path,
                    'label': label,
                    'name': img_name,
                    'error': str(e)
                })
                print(f'❌ CORROMPIDA: {img_name}')
                print(f'   Erro: {e}')
    
    # Resumo
    print('\n' + '='*60)
    print('RESUMO DA VERIFICAÇÃO')
    print('='*60)
    print(f'✅ Imagens válidas: {len(valid_images)}')
    print(f'❌ Imagens corrompidas: {len(corrupted_images)}')
    
    if corrupted_images:
        print('\n' + '='*60)
        print('IMAGENS CORROMPIDAS ENCONTRADAS:')
        print('='*60)
        for img in corrupted_images:
            print(f'\n📍 {img["label"]}/{img["name"]}')
            print(f'   Caminho: {img["path"]}')
            print(f'   Erro: {img["error"]}')
        
        # Salvar lista de imagens corrompidas
        report_path = './data/corrupted_images_report.json'
        with open(report_path, 'w') as f:
            json.dump(corrupted_images, f, indent=2)
        
        print(f'\n📄 Relatório salvo em: {report_path}')
        
        # Criar script para remover imagens corrompidas
        remove_script = './data/remove_corrupted_images.sh'
        with open(remove_script, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# Script para remover imagens corrompidas\n\n')
            for img in corrupted_images:
                f.write(f'rm "{img["path"]}"\n')
        
        print(f'📄 Script de remoção criado: {remove_script}')
        print('\n💡 Para remover as imagens corrompidas:')
        print(f'   bash {remove_script}')
        print('   OU delete manualmente os arquivos listados acima')
    else:
        print('\n✨ Nenhuma imagem corrompida encontrada!')
    
    return valid_images, corrupted_images

if __name__ == '__main__':
    valid, corrupted = check_dataset()
    
    if corrupted:
        print('\n' + '='*60)
        print('⚠️  ATENÇÃO: Encontradas imagens corrompidas!')
        print('='*60)
        print('Recomendações:')
        print('1. Revise a lista de imagens corrompidas')
        print('2. Tente baixar o dataset novamente')
        print('3. OU remova as imagens corrompidas antes de treinar')
    else:
        print('\n' + '='*60)
        print('✅ Dataset OK! Pronto para treinamento.')
        print('='*60)
