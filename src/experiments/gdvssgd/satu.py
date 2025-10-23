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
learning_rate = 0.1  # Fixed for both GD and SGD
n_epochs = 20
batch_size_sgd = 1

# --- 4. Gradient Descent (GD) ---
print("Training Gradient Descent...")
theta_gd = initial_theta.copy()
gd_loss_history = []

for epoch in range(n_epochs):
    y_hat = sigmoid(X_b.dot(theta_gd))
    gradient = (1/m) * X_b.T.dot(y_hat - y)
    theta_gd -= learning_rate * gradient
    gd_loss_history.append(compute_loss(X_b, y, theta_gd))

# --- 5. Stochastic Gradient Descent (SGD) ---
print("Training Stochastic Gradient Descent...")
theta_sgd = initial_theta.copy()
sgd_loss_history = []

for epoch in range(n_epochs):
    for i in range(m // batch_size_sgd):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + batch_size_sgd]
        yi = y[random_index:random_index + batch_size_sgd]
        y_hat = sigmoid(xi.dot(theta_sgd))
        gradient_sgd = (1/batch_size_sgd) * xi.T.dot(y_hat - yi)
        theta_sgd -= learning_rate * gradient_sgd
    sgd_loss_history.append(compute_loss(X_b, y, theta_sgd))

# --- 6. Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), gd_loss_history, 'b-', label='Gradient Descent')
plt.plot(range(1, n_epochs + 1), sgd_loss_history, 'r--', label='Stochastic Gradient Descent')
plt.title('GD vs SGD Loss Trajectory (MNIST Subset)')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)

# Save plot and results
script_dir = Path("src/experiments/gdvssgd/results")  # Use current working directory
out_dir = script_dir
out_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(out_dir / "gd_vs_sgd_loss.png", bbox_inches="tight", dpi=200)
plt.savefig(out_dir / "gd_vs_sgd_loss.pdf", bbox_inches="tight")
plt.close()

# Save loss history
loss_df = pd.DataFrame({
    "epoch": range(1, n_epochs + 1),
    "gd_loss": gd_loss_history,
    "sgd_loss": sgd_loss_history
})
loss_df.to_csv(out_dir / "loss_history.csv", index=False)

# Save theta arrays
np.save(out_dir / "theta_gd.npy", theta_gd)
np.save(out_dir / "theta_sgd.npy", theta_sgd)
np.savetxt(out_dir / "theta_gd.csv", theta_gd.flatten(), delimiter=",")
np.savetxt(out_dir / "theta_sgd.csv", theta_sgd.flatten(), delimiter=",")

print(f"Saved plot: {out_dir / 'gd_vs_sgd_loss.png'}")
print(f"Saved plot (pdf): {out_dir / 'gd_vs_sgd_loss.pdf'}")
print(f"Saved loss history: {out_dir / 'loss_history.csv'}")
print(f"Saved theta arrays: {out_dir / 'theta_gd.npy'}, {out_dir / 'theta_sgd.npy'}")