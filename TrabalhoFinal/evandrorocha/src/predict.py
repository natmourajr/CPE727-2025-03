"""
Script para fazer predições usando modelos treinados
"""
import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

from models import create_model


class TBPredictor:
    """Classe para fazer predições de tuberculose"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Caminho para o arquivo .pth do modelo
            device: Dispositivo ('cuda' ou 'cpu')
        """
        self.device = device
        
        # Carregar checkpoint
        print(f'Carregando modelo de: {model_path}')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extrair informações do checkpoint
        model_name = checkpoint.get('model_name', 'resnet50')
        
        # Criar modelo
        self.model = create_model(
            model_name=model_name,
            pretrained=False,  # Não precisamos dos pesos pré-treinados
            num_classes=2,
            dropout=0.5
        )
        
        # Carregar pesos treinados
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f'Modelo {model_name} carregado com sucesso!')
        print(f'Época: {checkpoint["epoch"]}')
        print(f'Métricas de validação:')
        for key, value in checkpoint['metrics'].items():
            if key != 'confusion_matrix':
                print(f'  {key}: {value:.4f}')
        
        # Transformações para as imagens
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Converter para 3 canais
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path: str) -> dict:
        """
        Faz predição para uma única imagem
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Dicionário com predição e probabilidades
        """
        # Carregar e preprocessar imagem
        image = Image.open(image_path).convert('L')  # Converter para escala de cinza
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Fazer predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Mapear classe para label
        class_names = {0: 'Normal', 1: 'Tuberculose'}
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Normal': probabilities[0][0].item(),
                'Tuberculose': probabilities[0][1].item()
            }
        }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Faz predições para múltiplas imagens
        
        Args:
            image_paths: Lista de caminhos para imagens
            
        Returns:
            Lista de dicionários com predições
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def main():
    """Função principal para fazer predições"""
    
    parser = argparse.ArgumentParser(description='Fazer predições de tuberculose')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Caminho para o arquivo .pth do modelo')
    parser.add_argument('--image', type=str,
                        help='Caminho para uma única imagem')
    parser.add_argument('--image-dir', type=str,
                        help='Diretório com múltiplas imagens')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Dispositivo (cuda ou cpu)')
    
    args = parser.parse_args()
    
    # Verificar device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Criar preditor
    predictor = TBPredictor(args.model_path, device=device)
    
    # Fazer predições
    if args.image:
        # Predição única
        print(f'\n{"="*60}')
        print(f'Analisando imagem: {args.image}')
        print(f'{"="*60}')
        
        result = predictor.predict_image(args.image)
        
        print(f'\n🔍 Resultado da Predição:')
        print(f'  Classe predita: {result["predicted_label"]}')
        print(f'  Confiança: {result["confidence"]*100:.2f}%')
        print(f'\n📊 Probabilidades:')
        print(f'  Normal: {result["probabilities"]["Normal"]*100:.2f}%')
        print(f'  Tuberculose: {result["probabilities"]["Tuberculose"]*100:.2f}%')
        
    elif args.image_dir:
        # Predições em lote
        import glob
        image_paths = glob.glob(os.path.join(args.image_dir, '*.png'))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '*.jpg')))
        
        print(f'\n{"="*60}')
        print(f'Analisando {len(image_paths)} imagens do diretório: {args.image_dir}')
        print(f'{"="*60}\n')
        
        results = predictor.predict_batch(image_paths)
        
        # Estatísticas
        tb_count = sum(1 for r in results if r['predicted_label'] == 'Tuberculose')
        normal_count = len(results) - tb_count
        
        print(f'\n📊 Resumo:')
        print(f'  Total de imagens: {len(results)}')
        print(f'  Normal: {normal_count} ({normal_count/len(results)*100:.1f}%)')
        print(f'  Tuberculose: {tb_count} ({tb_count/len(results)*100:.1f}%)')
        
        # Mostrar resultados individuais
        print(f'\n📋 Resultados Individuais:')
        for result in results:
            filename = os.path.basename(result['image_path'])
            label = result['predicted_label']
            conf = result['confidence'] * 100
            emoji = '🔴' if label == 'Tuberculose' else '🟢'
            print(f'  {emoji} {filename:40s} -> {label:15s} ({conf:.1f}%)')
    
    else:
        print('Erro: Especifique --image ou --image-dir')


if __name__ == '__main__':
    main()
