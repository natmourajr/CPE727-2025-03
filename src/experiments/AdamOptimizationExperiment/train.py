

import os
import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# demo_adam_variants.py
# Adam, AdamW, AMSGrad, RAdam on Rosenbrock; improved plotting and outputs.


# Reproducibility and device
# ----------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


# Rosenbrock function (a=1, b=100)
# f(x,y) = (1-x)^2 + 100*(y - x^2)^2
# ----------------------------
def rosenbrock(theta: torch.Tensor) -> torch.Tensor:
    x, y = theta[0], theta[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

def rosenbrock_grid(xmin=-2, xmax=2, ymin=-1, ymax=3, n=400):
    xs = torch.linspace(xmin, xmax, n)
    ys = torch.linspace(ymin, ymax, n)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    return X.numpy(), Y.numpy(), Z.numpy()


# Optimization runner for 2D parameters
# ----------------------------
def run_optimizer(opt_ctor, steps=800, lr=3e-3, init=(-1.5, 2.0), **opt_kwargs):
    theta = nn.Parameter(torch.tensor(init, dtype=torch.float32, device=device))
    opt = opt_ctor([theta], lr=lr, **opt_kwargs)

    traj, losses, step_norms = [], [], []
    prev = theta.detach().clone()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = rosenbrock(theta)
        loss.backward()
        opt.step()

        with torch.no_grad():
            traj.append(theta.detach().cpu().numpy().copy())
            losses.append(float(loss.item()))
            step_norms.append(float(torch.norm(theta - prev).item()))
            prev.copy_(theta)

    return np.array(traj), np.array(losses), np.array(step_norms)


# Utilities
# ----------------------------
def smooth(y, k=7):
    """Simple centered moving average; k auto-limited for short series."""
    if k <= 1 or len(y) < 3:
        return y
    k = min(k, max(3, len(y) // 20))
    w = np.ones(k, dtype=float) / k
    return np.convolve(y, w, mode="same")


# Main experiment and plotting
# ----------------------------
def experiment_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="logs", help="folder to save figures")
    parser.add_argument("--steps", type=int, default=800, help="iterations per run")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--show", action="store_true", help="also show figures interactively")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    steps = args.steps
    lr = args.lr
    init = (-1.5, 2.0)


    # Baseline variants
    # ----------------------------
    variants = [
        ("Adam",    torch.optim.Adam,  {"amsgrad": False}),
        ("AMSGrad", torch.optim.Adam,  {"amsgrad": True}),
        ("RAdam",   torch.optim.RAdam, {}),
        ("AdamW",   torch.optim.AdamW, {}),
    ]

    results = {}
    for name, ctor, kw in variants:
        traj, loss, dth = run_optimizer(ctor, steps=steps, lr=lr, init=init, **kw)
        results[name] = {"traj": traj, "losses": loss, "dtheta": dth}

    # Adam (coupled L2) vs AdamW (decoupled)
    # ---------------------------------------
    _, loss_adam_l2, _ = run_optimizer(torch.optim.Adam,  steps=steps, lr=lr, init=init, weight_decay=1e-2)
    _, loss_adamw,   _ = run_optimizer(torch.optim.AdamW, steps=steps, lr=lr, init=init, weight_decay=1e-2)

    # Adam hyperparameter sweep (beta2 and eps)
    # ------------------------------------------
    sweep_specs = [
        ("Adam beta2=0.99", {"betas": (0.9, 0.99)}),
        ("Adam beta2=0.95", {"betas": (0.9, 0.95)}),
        ("Adam eps=1e-7",   {"eps": 1e-7}),
        ("Adam eps=1e-5",   {"eps": 1e-5}),
    ]
    sweep_curves = []
    for label, kw in sweep_specs:
        _, s_loss, _ = run_optimizer(torch.optim.Adam, steps=steps, lr=lr, init=init, **kw)
        sweep_curves.append((label, s_loss))

    # Plots
    # ----------------------------
    X, Y, Z = rosenbrock_grid()

    # 1) Trajectories
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = plt.gca()
    cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 24), linewidths=0.6)
    ax.clabel(cs, inline=1, fontsize=7)

    all_pts = np.concatenate([results[n]["traj"] for n in results], axis=0)
    pad = 0.10
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")

    for name in results:
        T = results[name]["traj"]
        ax.plot(T[:, 0], T[:, 1], label=name, linewidth=2)
        ax.plot(T[::50, 0], T[::50, 1], linestyle="none", marker=".", ms=4)
        ax.plot(T[0, 0], T[0, 1], "o", ms=5)
        ax.plot(T[-1, 0], T[-1, 1], "x", ms=6)

    ax.set_title("Rosenbrock: parameter-space trajectories (zoomed)")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "traj_contours_v2.png"), dpi=240, bbox_inches="tight")

    # 2) Loss vs iterations
    all_losses = np.concatenate([d["losses"] for d in results.values()])

    ylo = np.percentile(all_losses, 5)
    yhi = np.percentile(all_losses, 98)

    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for name in results:
        y = results[name]["losses"]
        ax.semilogy(smooth(y, k=9), label=name, linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_ylim(max(1e-8, ylo), yhi)
    ax.set_title("Loss vs iterations (smoothed)")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curves_v2.png"), dpi=240, bbox_inches="tight")

    # Early-phase for loss
    T0 = min(120, steps)
    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for name in results:
        y = results[name]["losses"][:T0]
        ax.semilogy(smooth(y, k=5), label=name, linewidth=2)
    ax.set_xlabel("iteration (first 120)")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss (early training)")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curves_early.png"), dpi=240, bbox_inches="tight")

    # 3) Step size over time (symlog)
    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for name in results:
        y = results[name]["dtheta"]
        ax.plot(smooth(y, k=7), label=name, linewidth=2)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\Delta \theta_t\|$")
    ax.set_title("Step size over time (symlog)")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "step_norms_v2.png"), dpi=240, bbox_inches="tight")

    # Early-phase for step sizes
    T0 = min(60, steps)
    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for name in results:
        y = results[name]["dtheta"][:T0]
        ax.plot(smooth(y, k=5), label=name, linewidth=2)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("iteration (first 60)")
    ax.set_ylabel(r"$\|\Delta \theta_t\|$")
    ax.set_title("Step size (early phase)")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "step_norms_early.png"), dpi=240, bbox_inches="tight")

    # 4) Adam (coupled L2) vs AdamW (decoupled)
    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    ax.semilogy(loss_adam_l2, label="Adam (wd=0.01)", linewidth=2)
    ax.semilogy(loss_adamw,  label="AdamW (wd=0.01)", linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Coupled L2 (Adam) vs Decoupled (AdamW)")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "adam_vs_adamw_weight_decay_v2.png"), dpi=240, bbox_inches="tight")

    # Ratio plot to magnify differences
    fig = plt.figure(figsize=(8.0, 3.4))
    ax = plt.gca()
    eps = 1e-12
    ratio = (loss_adamw + eps) / (loss_adam_l2 + eps)
    ax.plot(smooth(ratio, k=11), linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("AdamW / Adam loss")
    ax.set_title("AdamW vs Adam — loss ratio")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "adam_vs_adamw_ratio.png"), dpi=240, bbox_inches="tight")

    # 5) Hyperparameter sweep: beta2 and eps in separate panels
    beta_curves = [(label, curve) for (label, curve) in sweep_curves if "beta2" in label]
    eps_curves  = [(label, curve) for (label, curve) in sweep_curves if "eps" in label]

    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for label, curve in beta_curves:
        ax.semilogy(smooth(curve, k=9), label=label, linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Adam hyperparameters: effect of beta2")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "adam_sweep_beta2.png"), dpi=240, bbox_inches="tight")

    fig = plt.figure(figsize=(8.0, 4.2))
    ax = plt.gca()
    for label, curve in eps_curves:
        ax.semilogy(smooth(curve, k=9), label=label, linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Adam hyperparameters: effect of eps")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(args.outdir, "adam_sweep_eps.png"), dpi=240, bbox_inches="tight")

    if args.show:
        plt.show()

if __name__ == "__main__":
    experiment_main()
