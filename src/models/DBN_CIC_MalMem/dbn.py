import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.RBM_CIC_MalMem.rbm import RBM 
import time

class DeepBeliefNetwork(nn.Module):
    """
    Deep Belief Network (DBN) - Uma pilha de RBMs para pré-treinamento.
    """
    def __init__(self, input_shape, num_classes, config, device='cuda'):
        super(DeepBeliefNetwork, self).__init__()
        
        self.device = device
        self.input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        
        # O DBN usa a mesma estrutura oculta do seu MLP/SAE:
        hidden_sizes = [256, 128, 64]
        self.rbm_layers = nn.ModuleList()
        
        visible_size = self.input_size
        
        # Cria a pilha de RBMs (rbm1: 16->256, rbm2: 256->128, rbm3: 128->64)
        for hidden_size in hidden_sizes:
            rbm = RBM(visible_size, hidden_size, device=device).to(device)
            self.rbm_layers.append(rbm)
            visible_size = hidden_size # A saída oculta torna-se a entrada da próxima RBM
            
        # Camada de Classificação (Softmax) - Usada apenas para fine-tuning, mas incluída aqui
        self.classifier_head = nn.Linear(hidden_sizes[-1], num_classes).to(device)

    def forward(self, x):
        """Passagem forward após o pré-treinamento, através da pilha RBM/Encoder."""
        # Usa o forward das RBMs (probabilidade de ativação)
        for rbm in self.rbm_layers:
            x = rbm.forward(x) 
        
        # Saída da Camada Latente (64) para o Classificador
        x = self.classifier_head(x)
        return x

    def pretrain(self, dataloader, num_epochs, learning_rate):
        """
        Pré-treinamento ganancioso (greedy layer-wise) de cada RBM.
        """
        print(f"\n--- INICIANDO PRÉ-TREINAMENTO GANANCIOSO DA DBN ---")
        
        # input_data armazenará o tensor completo da saída da camada anterior (CPU)
        input_data = None 
        all_rbm_loss_histories = []
        # Itera e treina CADA RBM na pilha
        for i, rbm in enumerate(self.rbm_layers):
            print(f"  > Treinando RBM {i+1}/{len(self.rbm_layers)}: {rbm.W.shape[0]} -> {rbm.W.shape[1]}")
            
            optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate)
            rbm_epoch_losses = []
            for epoch in range(1, num_epochs + 1):
                total_loss = 0
                
                # --- LOOP DE TREINAMENTO (Usando input_data ou dataloader) ---
                for batch_idx, (data, _) in enumerate(dataloader):
                    
                    if i == 0:
                        # RBM 1/3: Usa os dados crus do dataloader (16 features)
                        v0 = data.float().to(self.device)
                        #v0 = v0.bernoulli()
                    else:
                        # RBM > 1: Usa o novo dataset (input_data) gerado pela RBM anterior
                        start_idx = batch_idx * data.size(0)
                        end_idx = start_idx + data.size(0)
                        v0 = input_data[start_idx:end_idx].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Passo de Contrastive Divergence (CD-1)
                    v0_cd, h0, vk, hk = rbm.contrastive_divergence(v0, k=1)
                    
                    # --- CORREÇÃO DA FUNÇÃO DE PERDA: BCE para camadas probabilísticas ---
                    if i == 0:
                        # RBM 1/3: Entrada de features contínuas. Usa MSE.
                        loss = F.mse_loss(v0, vk) 
                    else:
                        # RBM 2/3 e 3/3: Entrada de probabilidades (saída Sigmoid). Usa BCE.
                        # vk é a probabilidade reconstruída p(v|h_k), v0 é a probabilidade de entrada.
                        loss = F.binary_cross_entropy(vk, v0)
                    # ---------------------------------------------------------------------
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * v0.size(0)

                avg_loss = total_loss / len(dataloader.dataset)
                rbm_epoch_losses.append(avg_loss)
                if (epoch % 10 == 0) or (epoch == num_epochs):
                    print(f"    Época {epoch}/{num_epochs} -> Perda Média de Reconstrução: {avg_loss:.4f}")
            all_rbm_loss_histories.append(rbm_epoch_losses)
            # --- TRANSIÇÃO ENTRE RBMs (CALCULA O NOVO INPUT_DATA) ---
            if i < len(self.rbm_layers) - 1:
                # Calcula e armazena as ativações de TODAS as amostras de uma vez
                rbm.eval()
                new_input_data_list = []
                
                with torch.no_grad():
                    
                    if i == 0:
                         # RBM 1/3: A entrada é o dataset original
                         current_tensor_source = dataloader.dataset.X.to(self.device)
                         
                    else:
                         # RBM > 1: A entrada é o input_data (saída da RBM anterior) que está na CPU
                         current_tensor_source = input_data.to(self.device)
                         
                    # Processa o tensor completo em minibatches
                    for batch_idx, (data, _) in enumerate(dataloader):
                         start_idx = batch_idx * data.size(0)
                         end_idx = start_idx + data.size(0)
                         
                         batch_v = current_tensor_source[start_idx:end_idx]
                         
                         # Gera a entrada para a próxima RBM (ativação média)
                         new_input_data_list.append(rbm.forward(batch_v).cpu())
                         
                input_data = torch.cat(new_input_data_list, dim=0) 
                rbm.train()
                print(f"  > RBM {i+1} concluída. Entrada para a próxima RBM pronta com shape: {input_data.shape}.")
                
        print("\n✅ Pré-treinamento da DBN concluído.")
        return all_rbm_loss_histories