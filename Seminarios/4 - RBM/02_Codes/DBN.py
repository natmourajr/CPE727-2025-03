from RBM import RBM

import pandas as pd
import torch
import torch.nn as nn


class DBN:
    """
    Class for implementing the Deep Belief Network (DBN) model.
    """
    def __init__(self, layers, lr, cd_k, category, relu = True):
        self.layers = layers
        self.relu = relu
        self.lr = lr
        self.cd_k = cd_k
        self.category = category

        self.rbms = [RBM(layers[l], layers[l + 1], category) for l in range(len(layers) - 1)]


    def pre_training(self, train_loader, epochs_per_layer = 5):
        details = []
        for l_idx, rbm in enumerate(self.rbms):

            training_loss = 0
            batches = 0
            print(f"RBM {l_idx + 1} Training:")
            for epoch in range(epochs_per_layer):
                for v_train, _ in train_loader:
                    v_train = v_train.view(v_train.size(0), -1)

                    with torch.no_grad():
                        h_train = v_train
                        for j in range(l_idx):
                            h_train = self.rbms[j].inference(h_train)

                    loss = rbm.contrastive_divergence(h_train)
                    training_loss += loss
                    batches += 1

                details.append({'rbm': l_idx, 'epoch': epoch + 1, 'loss': training_loss / batches})
                print(f"  Epoch {epoch + 1}/{epochs_per_layer} - loss: {training_loss / batches:.4f}")

        return pd.DataFrame(details)
    

    def get_reconstructions(self, v_input):
        v_input = v_input.view(v_input.size(0), -1)

        details = []
        for l_idx, rbm in enumerate(self.rbms):
            with torch.no_grad():
                h_recon = v_input
                for j in range(l_idx + 1):
                    h_recon = self.rbms[j].inference(h_recon)
                
                details.append(h_recon)

        return details
                

    def build_classifier(self, n_classes):
        layers = []

        for i, rbm in enumerate(self.rbms):
            linear = nn.Linear(rbm.n_visible, rbm.n_hidden)

            with torch.no_grad():
                linear.weight.copy_(rbm.W.t())
                linear.bias.copy_(rbm.c)

            layers.append(linear)
            if self.relu:
                layers.append(nn.ReLU())

        linear_class = nn.Linear(self.layers[-1], n_classes)
        layers.append(linear_class)

        model = nn.Sequential(*layers)

        return model