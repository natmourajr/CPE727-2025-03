"""
Script para gerar visualizações Grad-CAM para ResNet50
Mostra onde o modelo está "olhando" para tomar a decisão
Versão sem dependência de OpenCV (usa apenas PIL/Matplotlib)
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import TBClassifier, SimpleCNN

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESNET_PATH = Path('models/resnet50_best.pth')
SIMPLECNN_PATH = Path('models/simplecnn_best.pth')
DENSENET_PATH = Path('models/densenet121_best.pth')
EFFICIENTNET_PATH = Path('models/efficientnet_b0_best.pth')
DATA_DIR = Path('data/shenzhen')
OUTPUT_DIR = Path('results/gradcam')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling dos gradientes
        # Verificar dimensões para garantir compatibilidade
        if gradients is None or activations is None:
             raise RuntimeError("Gradientes ou ativações não foram capturados. Verifique o target_layer.")

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Combinação linear ponderada
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU e Normalização
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Calcular probabilidade real com Softmax
        probs = torch.nn.functional.softmax(output, dim=1)
        prob_score = probs[0, class_idx].item()
        
        return cam.data.cpu().numpy()[0, 0], class_idx, prob_score

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return img, tensor

def overlay_heatmap(img, mask, alpha=0.5, colormap='jet'):
    """Sobrepõe heatmap na imagem usando PIL/Matplotlib"""
    # 1. Redimensionar máscara para o tamanho da imagem
    mask_img = Image.fromarray(np.uint8(255 * mask))
    mask_img = mask_img.resize(img.size, resample=Image.BILINEAR)
    mask_resized = np.array(mask_img) / 255.0
    
    # 2. Aplicar colormap visual (jet)
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(mask_resized) # Retorna RGBA
    heatmap = np.delete(heatmap, 3, 2) # Remover canal Alpha
    
    # 3. Converter imagem original para array float [0,1]
    img_array = np.array(img) / 255.0
    
    # 4. Superpor
    overlay = heatmap * alpha + img_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    
    return np.uint8(255 * overlay), np.uint8(255 * heatmap)

def visualize_gradcam(model, target_layer, img_path, save_name, title_prefix="Grad-CAM"):
    img, tensor = preprocess_image(img_path)
    tensor = tensor.to(DEVICE)
    
    # Inicializar GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Gerar heatmap
    try:
        mask, class_idx, prob = grad_cam(tensor)
    except Exception as e:
        print(f"Erro ao gerar GradCAM para {save_name}: {e}")
        return

    # Gerar imagem com overlay
    overlay_img, heatmap_img = overlay_heatmap(img, mask)
    
    # Plotar
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    class_name = "TB" if class_idx == 1 else "Normal"
    prob_tb = prob if class_idx == 1 else 1-prob
    
    fig.suptitle(f'{title_prefix}\nPredição: {class_name} (Prob TB: {prob_tb:.1%})', 
                fontsize=16, fontweight='bold')
    
    ax1.imshow(img)
    ax1.set_title("Imagem Original")
    ax1.axis('off')
    
    ax2.imshow(heatmap_img)
    ax2.set_title("Heatmap de Atenção")
    ax2.axis('off')
    
    ax3.imshow(overlay_img)
    ax3.set_title("Sobreposição")
    ax3.axis('off')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grad-CAM salvo: {save_path}")
    plt.close()

def load_all_models():
    """Carrega todos os modelos na memória"""
    models = {}
    
    # 1. ResNet50
    try:
        model = TBClassifier('resnet50', num_classes=2, pretrained=False)
        if RESNET_PATH.exists():
            ckpt = torch.load(RESNET_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(DEVICE).eval()
            models['resnet50'] = model
    except Exception as e: print(f"Erro ResNet: {e}")

    # 2. SimpleCNN
    try:
        model = SimpleCNN(num_classes=2, dropout=0.4)
        if SIMPLECNN_PATH.exists():
            ckpt = torch.load(SIMPLECNN_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(DEVICE).eval()
            models['simplecnn'] = model
    except Exception as e: print(f"Erro SimpleCNN: {e}")

    # 3. DenseNet121
    try:
        model = TBClassifier('densenet121', num_classes=2, pretrained=False)
        if DENSENET_PATH.exists():
            ckpt = torch.load(DENSENET_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(DEVICE).eval()
            models['densenet121'] = model
    except Exception as e: print(f"Erro DenseNet: {e}")
    
    # 4. EfficientNet-B0
    try:
        model = TBClassifier('efficientnet_b0', num_classes=2, pretrained=False)
        if EFFICIENTNET_PATH.exists():
            ckpt = torch.load(EFFICIENTNET_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(DEVICE).eval()
            models['efficientnet_b0'] = model
    except Exception as e: print(f"Erro EfficientNet: {e}")
    
    return models

def predict_single(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        # USANDO SOFTMAX PARA 2 CLASSES!
        probs = torch.nn.functional.softmax(output, dim=1)
        prob_tb = probs[0, 1].item() # Probabilidade da classe 1 (TB)
        return prob_tb

def main():
    print("="*60)
    print("GERANDO GRAD-CAM COMPARATIVO (4 MODELOS)")
    print("="*60)
    
    models = load_all_models()
    if len(models) < 4:
        print(f"⚠️ Atenção: Apenas {len(models)} modelos carregados: {list(models.keys())}")
    
    # Encontrar uma imagem onde TODOS acertem ou tenham alta confiança
    tb_images = list((DATA_DIR / 'tuberculosis').glob('*.png'))
    best_img = None
    best_avg_prob = -1.0
    
    print(f"🔍 Buscando imagem de consenso (total {len(tb_images)})...")
    
    # Verificar as 20 primeiras imagens para não demorar demais
    for img_path in tb_images[:30]:
        img, tensor = preprocess_image(img_path)
        tensor = tensor.to(DEVICE)
        
        probs = []
        for name, model in models.items():
            p = predict_single(model, tensor)
            probs.append(p)
        
        # Critério: Pelo menos a ResNet e a Baseline devem prever TB (>0.5)
        # Idealmente todos > 0.5
        all_tb = all(p > 0.5 for p in probs)
        avg_prob = sum(probs) / len(probs)
        
        if all_tb:
            print(f"✅ FOUND CONSENSUS IMAGE: {img_path.name} | Probs: {[f'{p:.2f}' for p in probs]}")
            best_img = img_path
            break
        
        # Fallback: Guarad a melhor média se não achar consenso perfeito
        if avg_prob > best_avg_prob and probs[0] > 0.5: # Pelo menos ResNet tem que garantir
            best_avg_prob = avg_prob
            best_img = img_path
            
    if not best_img:
        print("❌ Não foi possível encontrar uma imagem de consenso TB perfeita. Usando a primeira.")
        best_img = tb_images[0]
        
    print(f"\n🖼️ Imagem Selecionada para Grad-CAM: {best_img.name}")
    
    # Gerar Visualization Para Cada Modelo
    
    # 1. ResNet50
    if 'resnet50' in models:
        visualize_gradcam(models['resnet50'], models['resnet50'].backbone.layer4, 
                         best_img, 'resnet50_gradcam_tb.png', "Grad-CAM ResNet50")

    # 2. SimpleCNN
    if 'simplecnn' in models:
        # conv4[3] é a última conv
        visualize_gradcam(models['simplecnn'], models['simplecnn'].conv4[3], 
                         best_img, 'simplecnn_gradcam_tb.png', "Grad-CAM SimpleCNN (Baseline)")

    # 3. DenseNet121
    if 'densenet121' in models:
        visualize_gradcam(models['densenet121'], models['densenet121'].backbone.features.denseblock4, 
                         best_img, 'densenet121_gradcam_tb.png', "Grad-CAM DenseNet121")

    # 4. EfficientNet-B0
    if 'efficientnet_b0' in models:
        visualize_gradcam(models['efficientnet_b0'], models['efficientnet_b0'].backbone.features[8], 
                         best_img, 'efficientnet_b0_gradcam_tb.png', "Grad-CAM EfficientNet-B0")

    print("\n✅ Todas as visualizações geradas para a imagem comum!")

if __name__ == '__main__':
    main()
