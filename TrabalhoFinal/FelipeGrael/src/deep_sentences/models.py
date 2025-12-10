import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, F1Score, PearsonCorrCoef
from typing import List


class RNNSiamese(L.LightningModule):

    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int = 300,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_fc_hidden: int = 128,
        rnn_type: str = 'lstm',
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        bidirectional: bool = True,
        padding_idx: int = 0,
        similarity_threshold: float = 3.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.padding_idx = padding_idx
        self.similarity_threshold = similarity_threshold

        self.embeddings = nn.Embedding(n_tokens, embedding_dim, padding_idx=padding_idx)

        if rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        elif rnn_type == 'gru':
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Must be 'lstm' or 'gru'.")

        self.encoder = rnn_cls(
            embedding_dim,
            n_hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        encoder_output_size = n_hidden * 2 if bidirectional else n_hidden

        self.fc = nn.Sequential(
            nn.Linear(encoder_output_size * 2, n_fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_hidden, 1),
        )

        self.train_accuracy = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.train_pearson = PearsonCorrCoef()
        self.val_accuracy = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_pearson = PearsonCorrCoef()

    def encode_sentence(self, x):
        embedded = self.embeddings(x)

        output, (hidden, cell) = self.encoder(embedded)

        if self.bidirectional:
            hidden = hidden.view(self.n_layers, 2, x.size(0), self.n_hidden)
            hidden = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
        else:
            hidden = hidden[-1]

        return hidden

    def forward(self, sentence1, sentence2):
        encoded1 = self.encode_sentence(sentence1)
        encoded2 = self.encode_sentence(sentence2)

        combined = torch.cat([encoded1, encoded2], dim=1)

        similarity = self.fc(combined)

        return similarity

    def training_step(self, batch, batch_idx):
        sentence1, sentence2, labels = batch

        predictions = self(sentence1, sentence2).squeeze()

        loss = F.mse_loss(predictions, labels.float())

        mae = F.l1_loss(predictions, labels.float())

        preds_binary = (predictions >= self.similarity_threshold).long()
        labels_binary = (labels >= self.similarity_threshold).long()

        acc = self.train_accuracy(preds_binary, labels_binary)
        f1 = self.train_f1(preds_binary, labels_binary)
        pearson = self.train_pearson(predictions, labels.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pearson', pearson, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sentence1, sentence2, labels = batch

        predictions = self(sentence1, sentence2).squeeze()

        loss = F.mse_loss(predictions, labels.float())

        mae = F.l1_loss(predictions, labels.float())

        preds_binary = (predictions >= self.similarity_threshold).long()
        labels_binary = (labels >= self.similarity_threshold).long()

        acc = self.val_accuracy(preds_binary, labels_binary)
        f1 = self.val_f1(preds_binary, labels_binary)
        pearson = self.val_pearson(predictions, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pearson', pearson, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


class CNNSiamese(L.LightningModule):

    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int = 300,
        kernel_sizes: List[int] = None,
        n_filters: int = 128,
        n_fc_hidden: int = 128,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        padding_idx: int = 0,
        similarity_threshold: float = 3.0,
        pooling_strategy: str = 'max',
    ):
        super().__init__()
        self.save_hyperparameters()

        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        if pooling_strategy not in ['max', 'mean', 'both']:
            raise ValueError(f"pooling_strategy must be 'max', 'mean', or 'both', got {pooling_strategy}")

        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.similarity_threshold = similarity_threshold
        self.pooling_strategy = pooling_strategy

        self.embeddings = nn.Embedding(n_tokens, embedding_dim, padding_idx=padding_idx)

        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=n_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Output size after concatenating all conv outputs
        # If 'both', we concatenate max and mean, doubling the size
        pooling_multiplier = 2 if pooling_strategy == 'both' else 1
        encoder_output_size = n_filters * len(kernel_sizes) * pooling_multiplier

        self.fc = nn.Sequential(
            nn.Linear(encoder_output_size * 2, n_fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_fc_hidden, 1),
        )

        self.train_accuracy = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.train_pearson = PearsonCorrCoef()
        self.val_accuracy = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_pearson = PearsonCorrCoef()

    def encode_sentence(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embeddings(x)  # (batch_size, seq_len, embedding_dim)

        # Conv1D expects (batch_size, channels, seq_len)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # Apply each convolution and pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, n_filters, seq_len - kernel_size + 1)

            if self.pooling_strategy == 'max':
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_outputs.append(pooled)
            elif self.pooling_strategy == 'mean':
                pooled = F.avg_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_outputs.append(pooled)
            elif self.pooling_strategy == 'both':
                max_pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                mean_pooled = F.avg_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_outputs.append(torch.cat([max_pooled, mean_pooled], dim=1))

        # Concatenate all pooled outputs
        encoded = torch.cat(conv_outputs, dim=1)
        encoded = self.dropout(encoded)

        return encoded

    def forward(self, sentence1, sentence2):
        encoded1 = self.encode_sentence(sentence1)
        encoded2 = self.encode_sentence(sentence2)

        combined = torch.cat([encoded1, encoded2], dim=1)

        similarity = self.fc(combined)

        return similarity

    def training_step(self, batch, batch_idx):
        sentence1, sentence2, labels = batch

        predictions = self(sentence1, sentence2).squeeze()

        loss = F.mse_loss(predictions, labels.float())

        mae = F.l1_loss(predictions, labels.float())

        preds_binary = (predictions >= self.similarity_threshold).long()
        labels_binary = (labels >= self.similarity_threshold).long()

        acc = self.train_accuracy(preds_binary, labels_binary)
        f1 = self.train_f1(preds_binary, labels_binary)
        pearson = self.train_pearson(predictions, labels.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pearson', pearson, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sentence1, sentence2, labels = batch

        predictions = self(sentence1, sentence2).squeeze()

        loss = F.mse_loss(predictions, labels.float())

        mae = F.l1_loss(predictions, labels.float())

        preds_binary = (predictions >= self.similarity_threshold).long()
        labels_binary = (labels >= self.similarity_threshold).long()

        acc = self.val_accuracy(preds_binary, labels_binary)
        f1 = self.val_f1(preds_binary, labels_binary)
        pearson = self.val_pearson(predictions, labels.float())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pearson', pearson, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
