# L-BFGS m-variation experiment (PyTorch)

import time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------
# Reprodutibilidade e device
# ----------------------------
SEED = 123
random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ----------------------------
# Dados: MNIST (subconjunto)
# ----------------------------
tfm = transforms.Compose([
    transforms.ToTensor(),
    # Normalização leve para estabilidade do LS
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
test_full  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)

# Subconjuntos para acelerar
N_TRAIN = 10_000
N_VAL   = 2_000
train_idx = list(range(N_TRAIN))
val_idx   = list(range(N_VAL))
train_ds = Subset(train_full, train_idx)
val_ds   = Subset(test_full,  val_idx)

# DataLoaders (apenas para iterar; L-BFGS acumula grad sobre TODO o loader)
BATCH = 512
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, drop_last=False)

# ----------------------------
# Modelo: MLP suave (Softplus)
# ----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.act = nn.Softplus(beta=1.0)
    def forward(self, x):
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# Avaliação (acurácia e loss)
# ----------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='sum')
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum/total, correct/total

# ----------------------------
# Rotina: treinar com L-BFGS
# ----------------------------
def train_lbfgs(history_size=10, outer_steps=15, inner_max_iter=10, lr=1.0):
    model = MLP().to(device)
    # Weight decay 0 (clássico para comparar curvatura)
    optim = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=inner_max_iter,           # iterações internas do line search por step()
        history_size=history_size,
        line_search_fn='strong_wolfe'
    )

    # Contadores e logs
    closure_calls = 0
    outer_losses = []
    t0 = time.time()

    N = len(train_ds)  # para obter grad da média (não soma)
    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optim.zero_grad(set_to_none=True)

        total_loss = 0.0
        # Acumula gradiente da MÉDIA: cada loss_batch é dividido por N
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_batch = F.cross_entropy(logits, y, reduction='sum') / N
            loss_batch.backward()  # acumula grad
            total_loss += loss_batch.item()

        # Retorna a perda média em TODO o conjunto
        return torch.tensor(total_loss, dtype=torch.float32, device=device)

    # Loop de passos "externos" de L-BFGS
    for step in range(outer_steps):
        loss = optim.step(closure)  # executa line search interno
        # Log da perda (como número Python)
        outer_losses.append(float(loss.detach().cpu()))
        # Pequeno relatório de progresso
        if (step+1) % 3 == 0 or step == 0:
            val_loss, val_acc = evaluate(model, val_loader)
            print(f"[m={history_size}] step {step+1:02d}/{outer_steps} "
                  f"| train_loss={outer_losses[-1]:.4f} "
                  f"| val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

    t1 = time.time()
    val_loss, val_acc = evaluate(model, val_loader)
    summary = {
        "m": history_size,
        "outer_steps": outer_steps,
        "inner_max_iter": inner_max_iter,
        "lr": lr,
        "final_train_loss": outer_losses[-1],
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "closure_calls": closure_calls,
        "wall_time_sec": t1 - t0,
        "loss_curve": outer_losses,
        "model": model,  # retorna caso queira inspecionar
    }
    return summary

# ----------------------------
# Rodar varredura em m
# ----------------------------
configs_m = [5, 10, 20, 50]
results = []
for m in configs_m:
    print("\n" + "="*70)
    print(f"Rodando L-BFGS com history_size m = {m}")
    res = train_lbfgs(history_size=m, outer_steps=15, inner_max_iter=10, lr=1.0)
    results.append(res)
    print(f"Concluído: m={m} | time={res['wall_time_sec']:.1f}s | "
          f"val_acc={res['final_val_acc']*100:.2f}% | closure_calls={res['closure_calls']}")

# ----------------------------
# Plot: Loss vs outer step
# ----------------------------
plt.figure(figsize=(8,5))
for res in results:
    y = res["loss_curve"]
    x = list(range(1, len(y)+1))
    plt.plot(x, y, marker='o', label=f"m={res['m']}")
plt.xlabel("L-BFGS outer step")
plt.ylabel("Train loss (mean over set)")
plt.title("L-BFGS: train loss vs outer step (MNIST subset)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nResumo por m")
print("-"*60)
for res in results:
    print(f"m={res['m']:>2} | time={res['wall_time_sec']:6.1f}s | "
          f"val_acc={res['final_val_acc']*100:6.2f}% | "
          f"val_loss={res['final_val_loss']:.4f} | "
          f"closure_calls={res['closure_calls']:4d}")
