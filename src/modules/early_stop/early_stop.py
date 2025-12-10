import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, min_mode: bool = True, path="", restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.min_mode = min_mode
        self.restore_best_weights = restore_best_weights

        self.path = path
        self.model_path = os.path.join(self.path, "best_model.pth")
        self.best_value_path = os.path.join(self.path, "early_stopping_best_value.txt")

        self.best_value = self.initialize_best_value()
        self.counter = 0
        self.should_stop = False

        print(f"Initial best value: {self.best_value}")

    def initialize_best_value(self):
        checkpoint_value = self.load_checkpoint()
        if checkpoint_value is not None:
            return checkpoint_value
        if self.min_mode == True:
            return float('inf')
        return float('-inf')

    def __call__(self, value, model):
        if self.compare_value_with_best_value(value):
            self.best_value = value
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights:
                self.restore(model)

    def compare_value_with_best_value(self, value):
        if self.min_mode and value < self.best_value - self.min_delta:
            return True
        if not self.min_mode and value > self.best_value + self.min_delta:
            return True
        return False
            

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.model_path)
        print(f"💾 Modelo salvo! Melhor value = {self.best_value:.4f}")

        with open(self.best_value_path, "w") as f:
            f.write(f"{self.best_value:.4f}")

    def load_checkpoint(self):
        if os.path.exists(self.best_value_path):
            with open(self.best_value_path, "r") as f:
                return float(f.read().strip())
        return None

    def restore(self, model):
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            print("♻️ Best weights restaurados (restore_best_weights=True)")
