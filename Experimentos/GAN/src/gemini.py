import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dimensões de Entrada
image_size = 32     # Dimensão das imagens CIFAR-10
nc = 3              # Número de canais de cor (RGB)
nz = 100            # Tamanho do vetor latente (ruído)
ngf = 64            # Tamanho dos mapas de características do Gerador
ndf = 64            # Tamanho dos mapas de características do Discriminador

# Hyperparâmetros de Treinamento
batch_size = 128
num_epochs = 50
lr = 0.0002         # Taxa de aprendizado (Learning Rate)
beta1 = 0.5         # Parâmetro para o otimizador Adam

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 1a camada (Entrada: nz x 1 x 1)
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), # Saída: (ngf*4) x 4 x 4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 2a camada
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # Saída: (ngf*2) x 8 x 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 3a camada
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # Saída: ngf x 16 x 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 4a camada (Saída: nc x 32 x 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # Usa Tanh para mapear a saída para o intervalo [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1a camada (Entrada: nc x 32 x 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # Saída: ndf x 16 x 16
            nn.LeakyReLU(0.2, inplace=True), # Usa LeakyReLU (melhora o treinamento da GAN)

            # 2a camada
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # Saída: (ndf*2) x 8 x 8
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3a camada
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # Saída: (ndf*4) x 4 x 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4a camada (Saída: 1 x 1 x 1)
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # Usa Sigmoid para mapear a saída para o intervalo [0, 1]
        )

    def forward(self, input):
        return self.main(input)
    
# Carregamento do dataset CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalização para o intervalo [-1, 1], compatível com a função Tanh do Gerador
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Instanciação dos modelos
netG = Generator().to(device)
netD = Discriminator().to(device)

# Aplicação da inicialização de pesos
netG.apply(weights_init)
netD.apply(weights_init)

# Função de Perda (Loss Function)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss (ideal para classificação binária 0/1)

# Vetor de Ruído Fixo para visualização do progresso
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Labels
real_label = 1.
fake_label = 0.

# Otimizadores
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print("Iniciando o Loop de Treinamento...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        ##############################################
        ### 1. Treinar D: Máxima Posição da Loss V(D, G)
        ##############################################

        ## Treinar com Imagens Reais
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass de imagens reais através do D
        output = netD(real_cpu).view(-1)
        # Calcula a perda de D em amostras reais
        errD_real = criterion(output, label)
        # Calcula gradientes
        errD_real.backward()
        D_x = output.mean().item()

        ## Treinar com Imagens Falsas
        # Gera o ruído
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Gera o lote de imagens falsas
        fake = netG(noise)
        label.fill_(fake_label) # Preenche o label com 0 (falso)
        
        # Classifica todas as imagens falsas com D
        output = netD(fake.detach()).view(-1)
        # Calcula a perda de D em amostras falsas
        errD_fake = criterion(output, label)
        # Calcula gradientes (adiciona aos gradientes das amostras reais)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Combina a perda total de D e atualiza os parâmetros de D
        errD = errD_real + errD_fake
        optimizerD.step()

        ##############################################
        ### 2. Treinar G: Mínima Posição da Loss V(D, G)
        ##############################################

        netG.zero_grad()
        label.fill_(real_label)  # Inverte os labels: Queremos que G faça D classificar as falsas como 1 (real)
        
        # Forward pass das falsas (Geradas anteriormente)
        output = netD(fake).view(-1)
        # Calcula a perda de G (baseada na classificação de D)
        errG = criterion(output, label)
        # Calcula gradientes
        errG.backward()
        D_G_z2 = output.mean().item()
        
        # Atualiza o Gerador
        optimizerG.step()
        
        # Impressão de status
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] \tLoss_D: {errD.item():.4f} \tLoss_G: {errG.item():.4f} \tD(x): {D_x:.4f} \tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    # Salvar Imagens de Progresso (opcional, mas recomendado)
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True),
                      f'./results/fake_samples_epoch_{epoch:03d}.png',
                      padding=2, normalize=True)