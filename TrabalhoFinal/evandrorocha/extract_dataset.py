import zipfile
import os
from pathlib import Path

print("=" * 50)
print("EXTRAINDO DATASET SHENZHEN")
print("=" * 50)

zip_path = Path("data/shenzhen_dataset.zip")
extract_path = Path("data/shenzhen")

if not zip_path.exists():
    print(f"[ERRO] Arquivo não encontrado: {zip_path}")
    exit(1)

print(f"\nArquivo: {zip_path}")
print(f"Tamanho: {zip_path.stat().st_size / (1024*1024):.1f} MB")
print(f"Destino: {extract_path}")

# Criar diretório de destino
extract_path.mkdir(parents=True, exist_ok=True)

print("\nExtraindo arquivos...")
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Listar alguns arquivos
        file_list = zip_ref.namelist()
        print(f"Total de arquivos no zip: {len(file_list)}")
        
        # Extrair todos os arquivos
        for i, file in enumerate(file_list):
            if i % 100 == 0:
                print(f"Progresso: {i}/{len(file_list)} arquivos...")
            zip_ref.extract(file, extract_path)
        
        print(f"Progresso: {len(file_list)}/{len(file_list)} arquivos...")
        
    print("\n" + "=" * 50)
    print("EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 50)
    
    # Verificar estrutura
    print("\nEstrutura do dataset:")
    for root, dirs, files in os.walk(extract_path):
        level = root.replace(str(extract_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:  # Mostrar apenas 2 níveis
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Mostrar apenas 5 arquivos por diretório
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... e mais {len(files) - 5} arquivos')
        if level >= 2:
            break
            
except Exception as e:
    print(f"\n[ERRO] Falha na extração: {e}")
    exit(1)
