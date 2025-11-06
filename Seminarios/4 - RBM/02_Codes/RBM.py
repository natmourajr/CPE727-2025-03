import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class RBM:
    """
    Class for implementing Restricted Boltzamnn Machine (RBM) model.
    """

    def __init__(self, n_visible, n_hidden, category, lr = 0.1, cd_k = 1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.cd_k = cd_k
        self.lr = lr

        self.W = nn.Parameter(torch.randn(self.n_visible, self.n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.n_visible))
        self.c = nn.Parameter(torch.zeros(self.n_hidden))

        self.category = category


    @torch.no_grad()
    def sample_h(self, v):
        prob_h = torch.sigmoid(self.c + v @ self.W)
        h = torch.bernoulli(prob_h)

        if self.category == 'b':
            return h
        elif self.category == 'p':
            return prob_h


    @torch.no_grad()
    def sample_v(self, h):
        prob_v = torch.sigmoid(self.b + h @ self.W.t())
        v = torch.bernoulli(prob_v)

        if self.category == 'b':
            return v
        elif self.category == 'p':
            return prob_v
        

    @torch.no_grad()
    def gibbs_sampling(self, v0):
        hk = self.sample_h(v0)

        for _ in range(self.cd_k):
            vk = self.sample_v(hk)
            hk = self.sample_h(vk)

        return vk, hk
    

    @torch.no_grad()
    def contrastive_divergence(self, v0):
        batch_size = v0.size(0)

        v0 = torch.reshape(v0, (batch_size, -1))

        if self.category == 'b':
            v0 = torch.bernoulli(v0)
        elif self.category == 'p':
            v0 = v0

        h0 = self.sample_h(v0)

        vk, hk = self.gibbs_sampling(v0)

        positive_phase = (v0.t() @ h0) / batch_size
        negative_phase = (vk.t() @ hk) / batch_size

        dW = self.lr * (positive_phase - negative_phase)
        db = self.lr * torch.mean((v0 - vk), dim=0)
        dc = self.lr * torch.mean((h0 - hk), dim=0)

        self.W += dW
        self.b += db
        self.c += dc

        loss = F.binary_cross_entropy(vk, v0)

        return loss.item()
    

    @torch.no_grad()
    def inference(self, v):
        v = v.view(v.size(0), -1)
        h = self.sample_h(v)

        return h
    

    @torch.no_grad()
    def generation(self, h):
        h = h.view(h.size(0), -1)
        v = self.sample_v(h)

        return v
    

    @torch.no_grad()
    def reconstruction(self, v):
        v = v.view(v.size(0), -1)
        h = self.sample_h(v)
        v_recon = self.sample_v(h)

        if self.category == 'b':
            v_recon = torch.bernoulli(v_recon)
        elif self.category == 'p':
            v_recon = v_recon

        return v_recon