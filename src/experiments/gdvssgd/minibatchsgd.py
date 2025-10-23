import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

# --- 1. Load and Preprocess MNIST Subset (Binary Classification 0 vs 1) ---
try:
    print("Fetching MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_full, y_full = mnist.data, mnist.target.astype(int)

    # Filter for classes 0 and 1
    mask = (y_full == 0) | (y_full == 1)
    X_binary = X_full[mask]
    y_binary = (y_full[mask] == 1).astype(int).reshape(-1, 1)

    # Take a small subset (500 samples)
    N_SAMPLES = 500
    X = X_binary[:N_SAMPLES]
    y = y_binary[:N_SAMPLES]

    print(f"Dataset size: {len(X)} samples.")

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Add bias term
    X_b = np.c_[np.ones((len(X), 1)), X]
    m, n_features = X_b.shape

except Exception as e:
    print(f"Could not fetch MNIST: {e}. Using synthetic data.")
    m, n_features = 500, 785
    X_b = np.random.rand(m, n_features)
    y = np.random.randint(0, 2, (m, 1))

# --- 2. Logistic Regression Functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X_b, y, theta):
    y_hat = sigmoid(X_b.dot(theta))
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# --- 3. Training Parameters ---
np.random.seed(42)
initial_theta = np.random.randn(n_features, 1) * 0.01
learning_rate = 0.1  # Fixed for both SGD variants
n_epochs = 20
batch_size_sgd = 1    # Standard SGD (single sample)
batch_size_mini = 32  # Mini-batch SGD

# --- 4. Standard SGD (Batch Size = 1) ---
print("Training Standard SGD...")
theta_sgd = initial_theta.copy()
sgd_loss_history = []

for epoch in range(n_epochs):
    # Shuffle indices for randomness
    indices = np.random.permutation(m)
    for i in range(0, m, batch_size_sgd):
        batch_indices = indices[i:i + batch_size_sgd]
        xi = X_b[batch_indices]
        yi = y[batch_indices]
        y_hat = sigmoid(xi.dot(theta_sgd))
        gradient = (1/batch_size_sgd) * xi.T.dot(y_hat - yi)
        theta_sgd -= learning_rate * gradient
    sgd_loss_history.append(compute_loss(X_b, y, theta_sgd))

# --- 5. Mini-batch SGD (Batch Size = 32) ---
print("Training Mini-batch SGD...")
theta_mini = initial_theta.copy()
mini_loss_history = []

for epoch in range(n_epochs):
    # Shuffle indices for randomness
    indices = np.random.permutation(m)
    for i in range(0, m, batch_size_mini):
        batch_indices = indices[i:i + batch_size_mini]
        xi = X_b[batch_indices]
        yi = y[batch_indices]
        y_hat = sigmoid(xi.dot(theta_mini))
        gradient = (1/batch_size_mini) * xi.T.dot(y_hat - yi)
        theta_mini -= learning_rate * gradient
    mini_loss_history.append(compute_loss(X_b, y, theta_mini))

# --- 6. Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), sgd_loss_history, 'r--', label='Standard SGD (Batch Size=1)')
plt.plot(range(1, n_epochs + 1), mini_loss_history, 'g-', label=f'Mini-batch SGD (Batch Size={batch_size_mini})')
plt.title('Standard SGD vs Mini-batch SGD Loss Trajectory (MNIST Subset)')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)

# Save plot and results
script_dir = Path("src/experiments/gdvssgd/results")  # Use current working directory
out_dir = script_dir
out_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(out_dir / "sgd_vs_mini_sgd_loss.png", bbox_inches="tight", dpi=200)
plt.savefig(out_dir / "sgd_vs_mini_sgd_loss.pdf", bbox_inches="tight")
plt.close()

# Save loss history
loss_df = pd.DataFrame({
    "epoch": range(1, n_epochs + 1),
    "sgd_loss": sgd_loss_history,
    "mini_sgd_loss": mini_loss_history
})
loss_df.to_csv(out_dir / "sgd_loss_history.csv", index=False)

# Save theta arrays
np.save(out_dir / "theta_sgd.npy", theta_sgd)
np.save(out_dir / "theta_mini_sgd.npy", theta_mini)
np.savetxt(out_dir / "theta_sgd.csv", theta_sgd.flatten(), delimiter=",")
np.savetxt(out_dir / "theta_mini_sgd.csv", theta_mini.flatten(), delimiter=",")

print(f"Saved plot: {out_dir / 'sgd_vs_mini_sgd_loss.png'}")
print(f"Saved plot (pdf): {out_dir / 'sgd_vs_mini_sgd_loss.pdf'}")
print(f"Saved loss history: {out_dir / 'sgd_loss_history.csv'}")
print(f"Saved theta arrays: {out_dir / 'theta_sgd.npy'}, {out_dir / 'theta_mini_sgd.npy'}")