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
batch_size = 1       # Single sample for both
momentum = 0.9       # Momentum coefficient for SGD with momentum

# --- 4. Standard SGD (Without Momentum) ---
print("Training Standard SGD (No Momentum)...")
theta_sgd = initial_theta.copy()
sgd_loss_history = []

for epoch in range(n_epochs):
    indices = np.random.permutation(m)
    for i in range(0, m, batch_size):
        batch_indices = indices[i:i + batch_size]
        xi = X_b[batch_indices]
        yi = y[batch_indices]
        y_hat = sigmoid(xi.dot(theta_sgd))
        gradient = (1/batch_size) * xi.T.dot(y_hat - yi)
        theta_sgd -= learning_rate * gradient
    sgd_loss_history.append(compute_loss(X_b, y, theta_sgd))

# --- 5. SGD with Momentum ---
print("Training SGD with Momentum...")
theta_momentum = initial_theta.copy()
momentum_loss_history = []
velocity = np.zeros_like(theta_momentum)  # Initialize velocity

for epoch in range(n_epochs):
    indices = np.random.permutation(m)
    for i in range(0, m, batch_size):
        batch_indices = indices[i:i + batch_size]
        xi = X_b[batch_indices]
        yi = y[batch_indices]
        y_hat = sigmoid(xi.dot(theta_momentum))
        gradient = (1/batch_size) * xi.T.dot(y_hat - yi)
        # Update velocity with momentum
        velocity = momentum * velocity - learning_rate * gradient
        theta_momentum += velocity  # Update parameters using velocity
    momentum_loss_history.append(compute_loss(X_b, y, theta_momentum))

# --- 6. Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), sgd_loss_history, 'r--', label='Standard SGD (No Momentum)')
plt.plot(range(1, n_epochs + 1), momentum_loss_history, 'g-', label=f'SGD with Momentum (μ={momentum})')
plt.title('Standard SGD vs SGD with Momentum Loss Trajectory (MNIST Subset)')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
# Save plot and results
script_dir = Path("src/experiments/gdvssgd/results")  # Use current working directory
out_dir = script_dir
out_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(out_dir / "sgd_vs_momentum_loss.png", bbox_inches="tight", dpi=200)
plt.savefig(out_dir / "sgd_vs_momentum_loss.pdf", bbox_inches="tight")
plt.close()

# Save loss history
loss_df = pd.DataFrame({
    "epoch": range(1, n_epochs + 1),
    "sgd_loss": sgd_loss_history,
    "momentum_loss": momentum_loss_history
})
loss_df.to_csv(out_dir / "sgd_momentum_loss_history.csv", index=False)

# Save theta arrays
np.save(out_dir / "theta_sgd.npy", theta_sgd)
np.save(out_dir / "theta_momentum.npy", theta_momentum)
np.savetxt(out_dir / "theta_sgd.csv", theta_sgd.flatten(), delimiter=",")
np.savetxt(out_dir / "theta_momentum.csv", theta_momentum.flatten(), delimiter=",")

print(f"Saved plot: {out_dir / 'sgd_vs_momentum_loss.png'}")
print(f"Saved plot (pdf): {out_dir / 'sgd_vs_momentum_loss.pdf'}")
print(f"Saved loss history: {out_dir / 'sgd_momentum_loss_history.csv'}")
print(f"Saved theta arrays: {out_dir / 'theta_sgd.npy'}, {out_dir / 'theta_momentum.npy'}")