import os, math, time, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../dataloaders/Cifar10Loader'))
from cifar_10_loader import LabelNoiseDataset
from cifar_10_loader import TestNoiseWrapper

#### Configuração ####
EPOCHS          = 10
BATCH_SIZE      = 128
NUM_WORKERS     = 2
SUBSET_TRAIN    = 10000
VAL_SPLIT       = 5000
LABEL_NOISE_P   = 0.2
SIGMAS_TEST     = (0.0, 0.05, 0.1, 0.2)
DEVICE = torch.device("cuda")

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def fmt_td(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def make_model(num_classes=10):
    m = tv.models.resnet18(num_classes=num_classes)
    return m

def make_optimizer(name, params):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=0.1, momentum=0.0, weight_decay=5e-4)
    if name == 'nesterov':
        return torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=1e-5)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
    if name == 'radam':
        return torch.optim.RAdam(params, lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)
    if name == 'l-bfgs':
        return torch.optim.LBFGS(params, lr=1.0, history_size=10, line_search_fn=None)
    raise ValueError(name)

def reset_peak_mem():
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

def peak_mem_mb():
    if DEVICE.type == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024**2)
    return float('nan')

#### Treinamento ####
def train_epoch(model, opt, loader, grad_noise_tau=0.0, is_lbfgs=False, pbar=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if is_lbfgs:
            def closure():
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                return loss
            loss = opt.step(closure)
            with torch.no_grad():
                out = model(x)
                pred = out.argmax(1)
                loss_val = F.cross_entropy(out, y).item()
        else:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            if grad_noise_tau > 0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * grad_noise_tau)
            opt.step()
            pred = out.argmax(1)
            loss_val = loss.item()

        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss_val * y.size(0)

        if pbar is not None:
            pbar.update(1)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, base_test_ds, sigmas=(0.0, 0.05, 0.1, 0.2), bs=256):
    model.eval()
    results = {}
    for s in sigmas:
        loader = cifar_10_loader(
            base_test_ds,
            sigma=s,
            batch_size=bs,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out, y, reduction='sum')
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += loss.item()
        results[s] = dict(acc=correct/total, loss=loss_sum/total)
    return results

def auc_robustez(acc_by_sigma: dict):
    xs = sorted(acc_by_sigma.keys())
    ys = [acc_by_sigma[x]['acc'] for x in xs]
    area = 0.0
    for i in range(len(xs)-1):
        dx = xs[i+1] - xs[i]
        area += 0.5 * (ys[i] + ys[i+1]) * dx
    return area

# LR search grid
LR_GRID = {
    'SGD':      0.1,
    'Nesterov': 0.1,
    'RMSProp':  1e-3,
    'AdamW':    3e-4,
    'RAdam':    3e-4,
    'L-BFGS':   1.0
}
OPTIMIZERS = ['SGD','Nesterov','RMSProp','AdamW','RAdam','L-BFGS']

#### Main pipeline ####
def main():
    logs = []
    global_start = time.perf_counter()

    print(f"Starting experiment on {DEVICE} | EPOCHS={EPOCHS} | BATCH_SIZE={BATCH_SIZE} | SUBSET_TRAIN={SUBSET_TRAIN}")
    print(f"Optimizers: {OPTIMIZERS}")

    p = LABEL_NOISE_P
    phase_start = time.perf_counter()

    train_part, val_part, test_set = build_datasets(
        data_root="./data",
        subset_train=SUBSET_TRAIN,
        p_label_noise=p,
        val_split=VAL_SPLIT,
        seed=SEED,
    )
    train_loader, val_loader = build_loaders(
        train_part,
        val_part,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ===== Treinamento final por otimizador com LR do grid =====
    print(f"\n[p={p}] Final training per optimizer with chosen LRs")
    for o_idx, opt_name in enumerate(OPTIMIZERS, start=1):
        print(f"  → [{o_idx}/{len(OPTIMIZERS)}] {opt_name} | LR={LR_GRID[opt_name]}")
        model = make_model(10).to(DEVICE)
        opt   = make_optimizer(opt_name, model.parameters())
        for g in opt.param_groups: g['lr'] = LR_GRID[opt_name]
        is_lbfgs = (opt_name.lower() == 'l-bfgs')

        reset_peak_mem()

        final_start = time.perf_counter()
        epoch_iter = range(1, EPOCHS+1)

        for ep in epoch_iter:
            tr_loss, tr_acc = train_epoch(
                model, opt, train_loader,
                grad_noise_tau=0,
                is_lbfgs=is_lbfgs,
                pbar=None
            )

        peak_mb = peak_mem_mb()
        eval_res = evaluate(model, test_set, sigmas=SIGMAS_TEST, bs=256)
        auc = auc_robustez(eval_res)
        elapsed_final = time.perf_counter() - final_start

        row = {
            'optimizer': opt_name,
            'label_noise_p': p,
            'Learning rate': LR_GRID[opt_name],
            'peak_mem_mb': peak_mb,
            'auc_robustez': auc
        }
        for s, d in eval_res.items():
            row[f'acc_sigma_{s}'] = d['acc']
            row[f'loss_sigma_{s}'] = d['loss']
        logs.append(row)

        accs = ", ".join([f"σ={s}: {eval_res[s]['acc']:.3f}" for s in SIGMAS_TEST])
        print(f"    Done {opt_name} | peak_mem={peak_mb:.1f} MB | AUC={auc:.4f} | {accs} | time={fmt_td(elapsed_final)}")

    phase_elapsed = time.perf_counter() - phase_start
    total_elapsed = time.perf_counter() - global_start
    print(f"\n=== Completed phase_time={fmt_td(phase_elapsed)}===\n")

    df = pd.DataFrame(logs)
    try:
        # Se estiver em notebook, display funciona; em script, apenas printa head
        display(df)
    except Exception:
        print(df)

    subset = df[df['label_noise_p'] == p]
    plt.figure(figsize=(6,4))
    for _, row in subset.iterrows():
        plt.scatter(row['auc_robustez'], row['peak_mem_mb'], s=60)
        plt.text(row['auc_robustez']*1.005, row['peak_mem_mb']*1.005, row['optimizer'])
    plt.ylabel('CUDA Peak Memory (MB)')
    plt.xlabel('Robustness AUC (Acc × σ)')
    plt.title(f'Pareto Memory × Robustness — p={p}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()
