"""
Script CLI principal para treinar e avaliar modelos
"""
import argparse
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.dirname(__file__))

from dataset import create_dataloaders
from models import create_model
from train import Trainer
from evaluate import ModelEvaluator
from utils import set_seed, get_device, print_model_summary
from config import DATA_CONFIG, TRAINING_CONFIG, MODEL_CONFIGS


def train_model(args):
    """Treina um modelo"""
    print("\n🚀 Iniciando treinamento...")
    print(f"Modelo: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device()
    
    # Criar dataloaders
    print("📊 Carregando dados...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    print(f"✅ Dados carregados:")
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    
    # Criar modelo
    print(f"\n🏗️  Criando modelo {args.model}...")
    model = create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        num_classes=2,
        dropout=args.dropout
    )
    
    print_model_summary(model)
    
    # Criar trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Treinar
    print("\n🎯 Iniciando treinamento...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Avaliar no teste
    print("\n📈 Avaliando no conjunto de teste...")
    test_metrics = trainer.validate(test_loader)
    
    print("\n" + "="*60)
    print("RESULTADOS FINAIS")
    print("="*60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print("="*60 + "\n")
    
    print(f"✅ Treinamento concluído! Modelo salvo em {args.save_dir}")


def evaluate_models(args):
    """Avalia e compara modelos"""
    print("\n📊 Iniciando avaliação de modelos...")
    
    device = get_device()
    
    # Criar dataloader de teste
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    # Criar avaliador
    evaluator = ModelEvaluator(device=device)
    
    # Avaliar modelos especificados
    models_evaluated = 0
    for model_name in args.models:
        model_path = os.path.join(args.model_dir, f'{model_name}_best.pth')
        
        if os.path.exists(model_path):
            print(f"\n✅ Avaliando {model_name}...")
            model = evaluator.load_model(model_path, model_name)
            evaluator.evaluate_model(model, test_loader, model_name)
            models_evaluated += 1
        else:
            print(f"⚠️  Modelo {model_name} não encontrado em {model_path}")
    
    if models_evaluated > 0:
        print(f"\n📈 Gerando comparações...")
        evaluator.plot_roc_curves(save_path=os.path.join(args.results_dir, 'roc_comparison.png'))
        evaluator.plot_pr_curves(save_path=os.path.join(args.results_dir, 'pr_comparison.png'))
        evaluator.compare_models()
        print(f"✅ Avaliação concluída! Resultados salvos em {args.results_dir}")
    else:
        print("❌ Nenhum modelo encontrado para avaliar")


def main():
    parser = argparse.ArgumentParser(description='Detecção de Tuberculose com Deep Learning')
    subparsers = parser.add_subparsers(dest='command', help='Comando a executar')
    
    # Subparser para treinar
    train_parser = subparsers.add_parser('train', help='Treinar um modelo')
    train_parser.add_argument('--model', type=str, default='resnet50',
                             choices=['resnet50', 'resnet101', 'densenet121', 
                                     'densenet169', 'efficientnet_b0', 'vgg16'],
                             help='Nome do modelo')
    train_parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                             help='Diretório dos dados')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Tamanho do batch')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Número de épocas')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5,
                             help='Weight decay (L2 regularization)')
    train_parser.add_argument('--dropout', type=float, default=0.5,
                             help='Dropout rate')
    train_parser.add_argument('--img-size', type=int, default=224,
                             help='Tamanho da imagem')
    train_parser.add_argument('--num-workers', type=int, default=4,
                             help='Número de workers para dataloader')
    train_parser.add_argument('--patience', type=int, default=10,
                             help='Paciência para early stopping')
    train_parser.add_argument('--pretrained', action='store_true', default=True,
                             help='Usar pesos pré-treinados')
    train_parser.add_argument('--save-dir', type=str, default='./models',
                             help='Diretório para salvar modelos')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Seed para reprodutibilidade')
    
    # Subparser para avaliar
    eval_parser = subparsers.add_parser('evaluate', help='Avaliar modelos')
    eval_parser.add_argument('--models', nargs='+', 
                            default=['resnet50', 'densenet121', 'efficientnet_b0'],
                            help='Modelos para avaliar')
    eval_parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                            help='Diretório dos dados')
    eval_parser.add_argument('--model-dir', type=str, default='./models',
                            help='Diretório dos modelos salvos')
    eval_parser.add_argument('--results-dir', type=str, default='./results',
                            help='Diretório para salvar resultados')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='Tamanho do batch')
    eval_parser.add_argument('--img-size', type=int, default=224,
                            help='Tamanho da imagem')
    eval_parser.add_argument('--num-workers', type=int, default=4,
                            help='Número de workers')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_models(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
