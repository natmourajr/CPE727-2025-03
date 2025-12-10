from dataloaders.image_dataloader import build_image_dataloaders
from models.cnn import ConvolutionalNeuralNetwork
from helpers. plot_cm import plot_confusion_matrix

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import os
import json
import torch
import random

from datetime import datetime
from torch import nn

from training.train_models import fit_model, get_test_predictions


train_loader, val_loader, test_loader = build_image_dataloaders(
                                            data_path = os.path.join(os.path.dirname(__file__), 'data', 'MT_dsa_history_data_with_labels.csv'),
                                            feature_exclude = ['StmTime', 'ThrCode', 'VltCode'],
                                            target_col = 'VltCode',
                                            batch_size = 64,
                                            train_size = 0.7,
                                            val_size = 0.15,
                                            random_state = random.randint(1, 100)
                                        )

# Models definition:
model = ConvolutionalNeuralNetwork(in_size=next(iter(train_loader))[0].shape[-1], channel1=16, channel2=23, num_l1=512, num_l2=64)

# Loss function and optimizer definition:
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Fitting model:
merit_figures = fit_model(train_loader, val_loader, model, criterion, optimizer, num_epochs = 40)

with open(os.path.join(os.path.dirname(__file__), 'history', f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_cnn_model_history.json"), "w", encoding="utf-8") as f:
    json.dump(merit_figures, f, ensure_ascii=False, indent=4)

# Evaluating test results:
y, y_pred = get_test_predictions(model, test_loader, threshold=0.5)

accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

cm = confusion_matrix(y, y_pred)

plot_confusion_matrix(cm, labels=["Normal", "Violado"], save_path=os.path.join(os.path.dirname(__file__), 'history', f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_cnn_model_cm.png"))