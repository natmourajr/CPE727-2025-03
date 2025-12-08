"""
Script para diagnosticar problemas no dataset e treinamento
"""
import os
import json
import torch
from dataset import ShenzhenTBDataset, create_dataloaders
from PIL import Image
import numpy as np


def check_dataset_distribution(data_dir: str):
    """Verifica a distribuição de classes no dataset"""
    print("="*60)
    print("DIAGNÓSTICO DO DATASET")
    print("="*60)
    
    # Contar imagens em cada diretório
    normal_dir = os.path.join(data_dir, 'normal')
    tb_dir = os.path.join(data_dir, 'tuberculosis')
    
    normal_count = len([f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(normal_dir) else 0
    tb_count = len([f for f in os.listdir(tb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(tb_dir) else 0
    
    total = normal_count + tb_count
    
    print(f"\n📊 Distribuição de Classes:")
    print(f"  Normal: {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"  Tuberculose: {tb_count} ({tb_count/total*100:.1f}%)")
    print(f"  Total: {total}")
    
    if normal_count != tb_count:
        ratio = max(normal_count, tb_count) / min(normal_count, tb_count)
        print(f"\n⚠️  DESBALANCEAMENTO DETECTADO!")
        print(f"  Razão: {ratio:.2f}:1")
        
        # Calcular class weights
        weight_normal = total / (2 * normal_count)
        weight_tb = total / (2 * tb_count)
        print(f"\n💡 Class Weights Recomendados:")
        print(f"  Normal: {weight_normal:.4f}")
        print(f"  Tuberculose: {weight_tb:.4f}")
    
    return normal_count, tb_count


def check_corrupted_images(data_dir: str):
    """Verifica imagens corrompidas"""
    print(f"\n{'='*60}")
    print("VERIFICAÇÃO DE IMAGENS CORROMPIDAS")
    print("="*60)
    
    corrupted = []
    
    for subdir in ['normal', 'tuberculosis']:
        dir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(dir_path):
            continue
            
        for img_name in os.listdir(dir_path):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(dir_path, img_name)
            try:
                img = Image.open(img_path)
                img.verify()  # Verifica integridade
                img = Image.open(img_path)  # Reabrir após verify
                img.load()  # Força carregar os dados
            except Exception as e:
                corrupted.append((img_path, str(e)))
                print(f"❌ {img_path}: {e}")
    
    if corrupted:
        print(f"\n⚠️  Encontradas {len(corrupted)} imagens corrompidas!")
        print("\n💡 Recomendação: Remover ou substituir essas imagens")
    else:
        print("\n✅ Nenhuma imagem corrompida encontrada!")
    
    return corrupted


def check_split_sizes(data_dir: str, batch_size: int = 16):
    """Verifica tamanhos dos splits"""
    print(f"\n{'='*60}")
    print("TAMANHOS DOS SPLITS")
    print("="*60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=(224, 224),
        num_workers=0  # Evitar problemas com multiprocessing
    )
    
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    total = train_size + val_size + test_size
    
    print(f"\n📦 Tamanhos:")
    print(f"  Train: {train_size} ({train_size/total*100:.1f}%) - {len(train_loader)} batches")
    print(f"  Val: {val_size} ({val_size/total*100:.1f}%) - {len(val_loader)} batches")
    print(f"  Test: {test_size} ({test_size/total*100:.1f}%) - {len(test_loader)} batches")
    
    if val_size < 50:
        print(f"\n⚠️  VALIDAÇÃO MUITO PEQUENA!")
        print(f"  Com apenas {val_size} amostras, as métricas serão instáveis")
        print(f"  Recomendação: Aumentar val_split ou usar cross-validation")
    
    return train_size, val_size, test_size


def analyze_training_history(history_path: str = './models/history.json'):
    """Analisa histórico de treinamento"""
    print(f"\n{'='*60}")
    print("ANÁLISE DO HISTÓRICO DE TREINAMENTO")
    print("="*60)
    
    if not os.path.exists(history_path):
        print(f"❌ Arquivo não encontrado: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = len(history['train_loss'])
    
    print(f"\n📈 Resumo:")
    print(f"  Épocas treinadas: {epochs}")
    print(f"  Train Loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc: {history['train_acc'][0]:.4f} -> {history['train_acc'][-1]:.4f}")
    print(f"  Val Loss: {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc: {history['val_acc'][0]:.4f} -> {history['val_acc'][-1]:.4f}")
    print(f"  Val F1: {history['val_f1'][0]:.4f} -> {history['val_f1'][-1]:.4f}")
    
    # Detectar overfitting
    train_acc_final = history['train_acc'][-1]
    val_acc_final = history['val_acc'][-1]
    gap = train_acc_final - val_acc_final
    
    print(f"\n🔍 Diagnóstico:")
    if gap > 0.2:
        print(f"  ❌ OVERFITTING SEVERO!")
        print(f"  Gap Train-Val: {gap*100:.1f}%")
        print(f"\n💡 Soluções:")
        print(f"  1. Aumentar dropout (atual: 0.5 -> tentar 0.6-0.7)")
        print(f"  2. Aumentar weight_decay (atual: 1e-5 -> tentar 1e-4)")
        print(f"  3. Usar mais data augmentation")
        print(f"  4. Congelar mais camadas da ResNet")
        print(f"  5. Reduzir learning rate")
    elif val_acc_final < 0.6:
        print(f"  ❌ MODELO NÃO ESTÁ APRENDENDO!")
        print(f"  Val Acc muito baixa: {val_acc_final*100:.1f}%")
        print(f"\n💡 Soluções:")
        print(f"  1. Verificar se as labels estão corretas")
        print(f"  2. Aumentar learning rate")
        print(f"  3. Treinar por mais épocas")
        print(f"  4. Usar modelo menor (EfficientNet-B0)")
    else:
        print(f"  ✅ Modelo parece estar aprendendo bem!")


def main():
    """Executa diagnóstico completo"""
    data_dir = './data/shenzhen'
    
    print("\n" + "="*60)
    print("DIAGNÓSTICO COMPLETO DO TREINAMENTO")
    print("="*60 + "\n")
    
    # 1. Verificar distribuição
    normal_count, tb_count = check_dataset_distribution(data_dir)
    
    # 2. Verificar imagens corrompidas
    corrupted = check_corrupted_images(data_dir)
    
    # 3. Verificar splits
    train_size, val_size, test_size = check_split_sizes(data_dir)
    
    # 4. Analisar histórico
    analyze_training_history()
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO E RECOMENDAÇÕES")
    print("="*60)
    
    print("\n🎯 Próximos Passos:")
    print("  1. Remover imagens corrompidas (se houver)")
    print("  2. Implementar class weights para desbalanceamento")
    print("  3. Ajustar hiperparâmetros (dropout, lr, weight_decay)")
    print("  4. Congelar camadas iniciais da ResNet (transfer learning)")
    print("  5. Re-treinar e validar resultados")


if __name__ == '__main__':
    main()
