import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    """
    Máquina de Boltzmann Restrita (RBM) para pré-treinamento do DBN.
    """
    def __init__(self, visible_size, hidden_size, device="cpu"):
        super(RBM, self).__init__()
        
        # Pesos (W) e Vieses (bias)
        self.W = nn.Parameter(torch.randn(visible_size, hidden_size) * 0.1) # Matriz de Conexão
        self.v_bias = nn.Parameter(torch.zeros(visible_size)) # Vieses da Camada Visível
        self.h_bias = nn.Parameter(torch.zeros(hidden_size))  # Vieses da Camada Oculta
        self.device = device
        
    def forward(self, v):
        """Calcula a probabilidade de ativação da camada oculta (p_h_dado_v)."""
        # p(h|v) = sigmoid(vW + h_bias)
        p_h_dado_v = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        return p_h_dado_v
        
    def sample_h_dado_v(self, v):
        """Amostra a camada oculta a partir da camada visível."""
        p_h_dado_v = self.forward(v)
        # Amostra com base na probabilidade de Bernoulli
        return p_h_dado_v.bernoulli() 

    def sample_v_dado_h(self, h):
        """Amostra a camada visível a partir da camada oculta (reconstrução)."""
        # p(v|h) = sigmoid(hW.t + v_bias)
        p_v_dado_h = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return p_v_dado_h
        
    def contrastive_divergence(self, v0, k=1):
        """
        Executa o passo de Contrastive Divergence (CD-k).
        v0: O dado de entrada (visível).
        k: Número de passos de Gibbs Sampling (geralmente k=1).
        """
        # h0 é a amostra da camada oculta a partir da entrada real (v0)
        h0 = self.sample_h_dado_v(v0)
        
        # Inicializa o Gibbs Sampling
        hk = h0
        
        # k passos de Gibbs Sampling
        for _ in range(k):
            # v <- h (Reconstrução)
            vk = self.sample_v_dado_h(hk)
            # h <- v (Amostra da oculta)
            hk = self.sample_h_dado_v(vk)
        
        # A derivada do CD é: (v0 * h0) - (vk * hk)
        # O gradiente é calculado com base na diferença esperada de ativações
        return v0, h0, vk, hk

    def get_code(self, v):
        """Método para obter a representação latente (código) após o treinamento."""
        # Retorna a probabilidade de ativação, não a amostra binária
        return self.forward(v)