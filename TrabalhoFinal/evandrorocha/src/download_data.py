"""
Script para download e preparação do dataset Shenzhen Hospital X-ray Set
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

# URLs do dataset
SHENZHEN_URL = "https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets"
# Nota: O download direto pode não funcionar devido a restrições do site NIH
# Mantenha as instruções manuais como alternativa
DATASET_ZIP_URL = "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"

def download_file(url, destination, resume=True):
    """
    Download de arquivo com barra de progresso e suporte a resumo
    
    Args:
        url: URL do arquivo
        destination: Path de destino
        resume: Se True, tenta continuar download parcial
    """
    destination = Path(destination)
    temp_file = destination.with_suffix(destination.suffix + '.part')
    
    # Verificar se há download parcial
    downloaded_size = 0
    if resume and temp_file.exists():
        downloaded_size = temp_file.stat().st_size
        print(f"📦 Download parcial encontrado: {downloaded_size / (1024*1024):.1f} MB")
        print("🔄 Retomando download...")
    
    headers = {}
    if downloaded_size > 0:
        headers['Range'] = f'bytes={downloaded_size}-'
    
    try:
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        
        # Verificar se servidor suporta range requests
        if downloaded_size > 0 and response.status_code not in [206, 200]:
            print("⚠️  Servidor não suporta resumo, baixando do início...")
            downloaded_size = 0
            headers = {}
            response = requests.get(url, stream=True, timeout=30, headers=headers)
        
        response.raise_for_status()
        
        # Tamanho total do arquivo
        if 'content-length' in response.headers:
            total_size = int(response.headers.get('content-length'))
            if response.status_code == 206:  # Partial content
                total_size = total_size + downloaded_size
        else:
            total_size = downloaded_size
        
        # Modo de abertura do arquivo
        mode = 'ab' if downloaded_size > 0 else 'wb'
        
        with open(temp_file, mode) as file, tqdm(
            desc=destination.name,
            total=total_size,
            initial=downloaded_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                progress_bar.update(size)
        
        # Mover arquivo completo para destino final
        temp_file.rename(destination)
        
        print("✅ Download concluído com sucesso!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro no download: {str(e)}")
        if temp_file.exists():
            print(f"💾 Download parcial salvo em: {temp_file}")
            print("🔄 Execute novamente para retomar o download")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Download interrompido pelo usuário")
        if temp_file.exists():
            print(f"💾 Download parcial salvo em: {temp_file}")
            print("🔄 Execute novamente para retomar o download")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {str(e)}")
        return False

def download_shenzhen_dataset(output_dir='./data'):
    """
    Download do dataset Shenzhen Hospital X-ray Set
    """
    print("=" * 70)
    print("DOWNLOAD DO DATASET SHENZHEN HOSPITAL X-RAY SET")
    print("=" * 70)
    
    # Criar diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Arquivo zip temporário
    zip_path = output_path / "shenzhen_dataset.zip"
    zip_part = output_path / "shenzhen_dataset.zip.part"
    
    # Verificar se já existe download completo
    if zip_path.exists():
        print(f"\n✅ Arquivo já existe: {zip_path}")
        print("📦 Pulando download e indo direto para extração...")
        success = True
    else:
        print("\n📥 Tentando baixar dataset automaticamente...")
        print(f"URL: {DATASET_ZIP_URL}")
        print(f"Destino: {zip_path}")
        
        if zip_part.exists():
            part_size = zip_part.stat().st_size
            print(f"\n🔄 Download parcial encontrado: {part_size / (1024*1024):.1f} MB")
            print("Tentando retomar download...\n")
        else:
            print("\n⚠️  Nota: O download automático pode falhar devido a restrições do site NIH.")
            print("Se falhar, siga as instruções de download manual abaixo.\n")
        
        # Tentar download automático com suporte a resumo
        success = download_file(DATASET_ZIP_URL, zip_path, resume=True)
    
    if not success or not zip_path.exists():
        print("\n" + "=" * 70)
        print("❌ DOWNLOAD AUTOMÁTICO FALHOU")
        print("=" * 70)
        print("\n📋 INSTRUÇÕES PARA DOWNLOAD MANUAL:\n")
        print("1. Acesse o site oficial:")
        print("   👉 https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets\n")
        print("2. Localize 'Shenzhen Hospital X-ray Set' e clique em 'Download'")
        print("3. O arquivo será: ChinaSet_AllFiles.zip (aproximadamente 440 MB)\n")
        print("4. Após baixar, coloque o arquivo .zip em:")
        print(f"   👉 {zip_path.absolute()}\n")
        print("5. Execute novamente este script para extrair e organizar:\n")
        print("   docker-compose run --rm tuberculosis-detection python src/download_data.py\n")
        print("=" * 70)
        return False
    
    # Se chegou aqui, o download foi bem-sucedido
    try:
        print("\n✅ Download concluído!")
        
        # Extrair arquivos
        print("\n📦 Extraindo arquivos...")
        extract_path = output_path / "shenzhen_raw"
        extract_path.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Extraindo"):
                zip_ref.extract(member, extract_path)
        
        print("✅ Extração concluída!")
        
        # Organizar dataset
        print("\n📂 Organizando dataset...")
        organize_dataset(extract_path, output_path / "shenzhen")
        
        # Limpar arquivos temporários
        print("\n🧹 Limpando arquivos temporários...")
        zip_path.unlink()
        shutil.rmtree(extract_path)
        
        print("\n✨ Dataset pronto para uso!")
        print(f"📍 Localização: {output_path / 'shenzhen'}")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante a extração/organização: {str(e)}")
        print("\nPor favor, verifique manualmente o arquivo baixado.")
        return False

def organize_dataset(source_dir, target_dir):
    """
    Organiza o dataset extraído em pastas normal/tuberculosis
    """
    target_path = Path(target_dir)
    source_path = Path(source_dir)
    
    # Criar estrutura de diretórios
    normal_dir = target_path / "normal"
    tb_dir = target_path / "tuberculosis"
    normal_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Procurando arquivos em: {source_path}")
    
    # O dataset Shenzhen geralmente vem com esta estrutura:
    # ChinaSet_AllFiles/
    #   ├── CXR_png/  (imagens)
    #   └── ClinicalReadings/ (metadados)
    
    # Procurar diretório de imagens
    image_dirs = list(source_path.rglob("**/CXR_png")) or \
                 list(source_path.rglob("**/images")) or \
                 [source_path]
    
    if image_dirs:
        image_dir = image_dirs[0]
        print(f"📸 Diretório de imagens encontrado: {image_dir}")
    else:
        image_dir = source_path
    
    # Procurar arquivo de metadados
    metadata_files = list(source_path.rglob("**/*ClinicalReadings*.txt")) or \
                     list(source_path.rglob("**/metadata*.txt")) or \
                     list(source_path.rglob("*.txt"))
    
    if metadata_files:
        print(f"📋 Arquivo de metadados encontrado: {metadata_files[0].name}")
        organize_from_metadata(image_dir, target_path, metadata_files[0])
    else:
        print("⚠️  Arquivo de metadados não encontrado.")
        print("📋 Organizando baseado na nomenclatura dos arquivos...")
        organize_by_filename(image_dir, target_path)
    
    # Contar imagens organizadas
    normal_count = len(list(normal_dir.glob("*.png")))
    tb_count = len(list(tb_dir.glob("*.png")))
    
    print(f"✅ Organizados: {normal_count} normais, {tb_count} com tuberculose")
    print(f"📊 Total: {normal_count + tb_count} imagens")

def organize_from_metadata(image_dir, target_dir, metadata_file):
    """
    Organiza dataset usando arquivo de metadados
    """
    normal_dir = target_dir / "normal"
    tb_dir = target_dir / "tuberculosis"
    
    # Ler arquivo de metadados
    try:
        with open(metadata_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # O formato típico inclui informações sobre cada imagem
        # Procurar padrões como "normal", "abnormal", "tuberculosis"
        for img_file in image_dir.glob("*.png"):
            filename = img_file.name
            
            # Verificar se o arquivo está mencionado nos metadados
            if filename in content:
                # Extrair a linha relevante
                for line in content.split('\n'):
                    if filename in line:
                        line_lower = line.lower()
                        if 'normal' in line_lower and 'abnormal' not in line_lower:
                            shutil.copy2(img_file, normal_dir / filename)
                        else:
                            # Assume tuberculose se não for normal
                            shutil.copy2(img_file, tb_dir / filename)
                        break
            else:
                # Se não encontrado nos metadados, usar nome do arquivo
                if 'normal' in filename.lower():
                    shutil.copy2(img_file, normal_dir / filename)
                else:
                    shutil.copy2(img_file, tb_dir / filename)
                    
    except Exception as e:
        print(f"⚠️  Erro ao processar metadados: {str(e)}")
        print("📋 Usando organização por nome de arquivo...")
        organize_by_filename(image_dir, target_dir)

def organize_by_filename(image_dir, target_dir):
    """
    Organiza dataset baseado no nome dos arquivos
    """
    normal_dir = target_dir / "normal"
    tb_dir = target_dir / "tuberculosis"
    
    # Procurar todas as imagens PNG
    for img_file in image_dir.rglob("*.png"):
        filename = img_file.name.lower()
        
        # Baseado na nomenclatura típica do dataset Shenzhen
        # Imagens normais geralmente contêm "normal" no nome
        # ou têm IDs específicos (CHNCXR_xxxx_0.png = normal, CHNCXR_xxxx_1.png = TB)
        if 'normal' in filename or filename.endswith('_0.png'):
            shutil.copy2(img_file, normal_dir / img_file.name)
        else:
            shutil.copy2(img_file, tb_dir / img_file.name)

def verify_dataset(data_dir='./data/shenzhen'):
    """
    Verifica a integridade do dataset baixado
    """
    data_path = Path(data_dir)
    
    print("\n" + "=" * 70)
    print("🔍 VERIFICAÇÃO DO DATASET")
    print("=" * 70)
    
    normal_dir = data_path / "normal"
    tb_dir = data_path / "tuberculosis"
    
    if not data_path.exists():
        print(f"❌ Diretório não encontrado: {data_path}")
        print("\n💡 Execute o download primeiro:")
        print("   docker-compose run --rm tuberculosis-detection python src/download_data.py")
        return False
    
    if not normal_dir.exists() or not tb_dir.exists():
        print("❌ Estrutura de diretórios incompleta!")
        print(f"\n📁 Estrutura esperada:")
        print(f"   {data_path}/")
        print(f"   ├── normal/")
        print(f"   └── tuberculosis/")
        return False
    
    normal_images = list(normal_dir.glob("*.png"))
    tb_images = list(tb_dir.glob("*.png"))
    
    print(f"\n📊 Estatísticas do Dataset:")
    print(f"   ✅ Imagens normais: {len(normal_images)}")
    print(f"   ✅ Imagens com TB: {len(tb_images)}")
    print(f"   📊 Total: {len(normal_images) + len(tb_images)} imagens")
    
    # Verificar números esperados (aproximados)
    expected_normal = 326
    expected_tb = 240
    total = len(normal_images) + len(tb_images)
    
    print(f"\n📈 Comparação com valores esperados:")
    print(f"   Normal: {len(normal_images)}/{expected_normal} ({len(normal_images)/expected_normal*100:.1f}%)")
    print(f"   TB: {len(tb_images)}/{expected_tb} ({len(tb_images)/expected_tb*100:.1f}%)")
    print(f"   Total: {total}/566 ({total/566*100:.1f}%)")
    
    if len(normal_images) == 0 or len(tb_images) == 0:
        print("\n⚠️  Dataset incompleto! Pelo menos uma categoria está vazia.")
        return False
    
    if total < 500:
        print("\n⚠️  Aviso: Número de imagens abaixo do esperado (566).")
        print("   Verifique se o download foi completo.")
    
    print("\n✅ Dataset verificado com sucesso!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download e preparação do dataset Shenzhen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Download e organização automática
  python src/download_data.py

  # Especificar diretório de saída
  python src/download_data.py --output-dir /caminho/personalizado

  # Apenas verificar dataset existente
  python src/download_data.py --verify-only

  # Organizar dataset baixado manualmente
  python src/download_data.py --organize-only --source /caminho/do/zip/extraido
        """
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Diretório de saída (padrão: ./data)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Apenas verificar dataset existente'
    )
    parser.add_argument(
        '--organize-only',
        action='store_true',
        help='Apenas organizar dataset já baixado'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Diretório fonte para organização (usar com --organize-only)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Limpar downloads parciais e recomeçar do zero'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forçar re-download mesmo se arquivo já existir'
    )
    
    args = parser.parse_args()
    
    # Limpar downloads parciais se solicitado
    if args.clean:
        output_path = Path(args.output_dir)
        zip_path = output_path / "shenzhen_dataset.zip"
        zip_part = output_path / "shenzhen_dataset.zip.part"
        
        print("🧹 Limpando downloads parciais...")
        
        if zip_part.exists():
            zip_part.unlink()
            print(f"✅ Removido: {zip_part}")
        
        if args.force and zip_path.exists():
            zip_path.unlink()
            print(f"✅ Removido: {zip_path}")
        
        if not zip_part.exists() and not (args.force and zip_path.exists()):
            print("ℹ️  Nenhum arquivo para limpar")
        
        print("✨ Limpeza concluída!")
        
        # Se apenas limpar, sair
        if not args.verify_only and not args.organize_only:
            print("\n💡 Execute novamente sem --clean para baixar o dataset")
            exit(0)
    
    if args.verify_only:
        verify_dataset(f"{args.output_dir}/shenzhen")
    elif args.organize_only:
        if not args.source:
            print("❌ Erro: --source é obrigatório quando usar --organize-only")
            exit(1)
        print(f"📂 Organizando dataset de: {args.source}")
        organize_dataset(args.source, f"{args.output_dir}/shenzhen")
        verify_dataset(f"{args.output_dir}/shenzhen")
    else:
        success = download_shenzhen_dataset(args.output_dir)
        if success:
            verify_dataset(f"{args.output_dir}/shenzhen")
