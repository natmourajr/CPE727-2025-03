from training.train_autoencoder import fit_autoencoder, fit_ae_dnn_classifier, get_test_predictions
from dataloaders.dataloader import build_dataloaders
from helpers.plot_cm import plot_confusion_matrix
from models.dnn import DeepNeuralNetwork
from models.ae import AutoEncoder

import os
import json
import torch

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from datetime import datetime
from torch import nn


train_loader, val_loader, test_loader, X_col = build_dataloaders(
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'MT_dsa_history_data_with_labels.csv'),
    feature_exclude = ['StmTime', 'ThrCode', 'VltCode'],
    target_col = 'VltCode',
    batch_size = 64,
    train_size = 0.7,
    val_size = 0.15,
    random_state = 13
)


# Autoencoder pre-training procedure:
ae = AutoEncoder(in_features=next(iter(train_loader))[0].shape[1], num_l1=256, num_l2=128, num_l3=32, ae_type='normal')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

history_ae = fit_autoencoder(train_loader, val_loader, ae, criterion, optimizer, num_epochs=40)

with open(os.path.join(os.path.dirname(__file__), 'history', f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_ae_model_history.json"), "w", encoding="utf-8") as f:
    json.dump(history_ae, f, ensure_ascii=False, indent=4)

# DNN training procedure:
dnn = DeepNeuralNetwork(in_features=32, num_l1=128, num_l2=64)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)

history_dnn = fit_ae_dnn_classifier(train_loader, val_loader, ae, dnn,criterion, optimizer, num_epochs=20)

with open(os.path.join(os.path.dirname(__file__), 'history', f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_ae_with_dnn_model_history.json"), "w", encoding="utf-8") as f:
    json.dump(history_dnn, f, ensure_ascii=False, indent=4)

y, y_pred = get_test_predictions(ae, dnn, test_loader, threshold=0.5)

accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

cm = confusion_matrix(y, y_pred)

plot_confusion_matrix(cm, labels=["Normal", "Violado"], save_path=os.path.join(os.path.dirname(__file__), 'history', f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_ae_with_dnn_model_cm.png"))