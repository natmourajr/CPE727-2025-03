import json
import os
import matplotlib.pyplot as plt


def load_history(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_history(history1, history2=None, figsize=(8, 4), savepath=None):

    plt.figure(figsize=figsize)

    plt.plot(range(1, len(history1['train_loss']) + 1), history1['train_loss'], label='CNN Train')
    plt.plot(range(1, len(history1['val_loss']) + 1), history1['val_loss'], label='CNN Validation')

    if history2:
        plt.plot(range(1, len(history2['train_loss']) + 1), history2['train_loss'], label='AE Train', linestyle='--')
        plt.plot(range(1, len(history2['val_loss']) + 1), history2['val_loss'], label='AE Validation', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.25)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()


def plot_recall_history(history1, history2=None, figsize=(8, 4), savepath=None):

    plt.figure(figsize=figsize)

    plt.plot(range(1, len(history1['train_recall']) + 1)[9:], history1['train_recall'][9:], label='CNN Train')
    plt.plot(range(1, len(history1['val_recall']) + 1)[9:], history1['val_recall'][9:], label='CNN Validation')

    if history2:
        plt.plot(range(1, len(history2['train_recall']) + 1)[9:], history2['train_recall'][9:], label='AE Train', linestyle='--')
        plt.plot(range(1, len(history2['val_recall']) + 1)[9:], history2['val_recall'][9:], label='AE Validation', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Training History")
    plt.grid(True, alpha=0.25)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":

    model1 = os.path.join(os.path.dirname(__file__), "2025_12_08_14_57_cnn_model_history_vlt.json")
    model2 = os.path.join(os.path.dirname(__file__), "2025_12_06_14_28_ae_model_history_vlt.json")

    hist1 = load_history(model1)
    hist2 = load_history(model2)

    plot_loss_history(hist1)
    plot_recall_history(hist1)