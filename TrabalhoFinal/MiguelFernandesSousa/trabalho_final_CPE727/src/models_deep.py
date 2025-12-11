"""
Deep Learning Models using PyTorch

This module provides CNN for Fashion MNIST and LSTM for AG News
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import DistilBertModel, DistilBertConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LeNet5Modified(nn.Module):
    """
    Modified LeNet-5 for Fashion MNIST (baseline rápido)

    Architecture:
    - Conv1: 1x28x28 -> 20x24x24 (5x5 kernel) -> MaxPool -> 20x12x12
    - Conv2: 20x12x12 -> 50x8x8 (5x5 kernel) -> MaxPool -> 50x4x4
    - FC1: 50*4*4 -> 500
    - FC2: 500 -> 300
    - FC3: 300 -> 10 classes

    Parameters: ~60K
    Expected accuracy: 90-91%
    Training time: ~2-3 min/época
    """

    def __init__(self, dropout=0.5):
        """
        Initialize LeNet-5 Modified

        Args:
            dropout: Dropout probability (default: 0.5)
        """
        super(LeNet5Modified, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 10)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 1, 28, 28]

        Returns:
            Output tensor [batch_size, 10]
        """
        # Conv1 + ReLU + Pool: (1, 28, 28) -> (20, 12, 12)
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 + ReLU + Pool: (20, 12, 12) -> (50, 4, 4)
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: (50, 4, 4) -> (800,)
        x = x.view(-1, 50 * 4 * 4)

        # FC1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # FC2 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc2(x)))

        # FC3 (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)

        return x


class ResNet18Adapted(nn.Module):
    """
    ResNet-18 adapted for Fashion MNIST 28x28

    Architecture: Simplified ResNet with residual blocks
    - Initial Conv: 1x28x28 -> 64x28x28 (3x3, padding=1)
    - Layer1: 2 residual blocks (64 channels)
    - Layer2: 2 residual blocks (128 channels, downsample)
    - Layer3: 2 residual blocks (256 channels, downsample)
    - Layer4: 2 residual blocks (512 channels, downsample)
    - AvgPool + FC: 512 -> 10

    Parameters: ~11M
    Expected accuracy: 94-95%
    Training time: ~8-10 min/época
    """

    class BasicBlock(nn.Module):
        """Basic residual block"""
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x):
            identity = x

            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = F.relu(out)

            return out

    def __init__(self, dropout=0.5):
        """
        Initialize ResNet-18 Adapted

        Args:
            dropout: Dropout probability (default: 0.5)
        """
        super(ResNet18Adapted, self).__init__()

        self.in_channels = 64

        # Initial convolution (no downsample for 28x28)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout and FC
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, out_channels, blocks, stride):
        """Create a residual layer"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(self.BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(self.BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 1, 28, 28]

        Returns:
            Output tensor [batch_size, 10]
        """
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout + FC
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MobileNetV2Small(nn.Module):
    """
    MobileNetV2 small for Fashion MNIST (eficiente)

    Architecture: Inverted residual blocks with depth multiplier 0.5
    - Initial Conv: 1x28x28 -> 16x28x28
    - Inverted residual blocks with depthwise separable convolutions
    - Final Conv: channels -> 640
    - AvgPool + FC: 640 -> 10

    Parameters: ~600K
    Expected accuracy: 93-94%
    Training time: ~5-7 min/época
    """

    class InvertedResidual(nn.Module):
        """Inverted residual block"""

        def __init__(self, in_channels, out_channels, stride, expand_ratio):
            super().__init__()
            self.stride = stride
            hidden_dim = int(in_channels * expand_ratio)
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

            layers = []
            if expand_ratio != 1:
                # Pointwise expansion
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                ])

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                         groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise projection
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ])

            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    def __init__(self, dropout=0.2, width_mult=0.5):
        """
        Initialize MobileNetV2 Small

        Args:
            dropout: Dropout probability (default: 0.2, lower for MobileNet)
            width_mult: Width multiplier (default: 0.5 for small variant)
        """
        super(MobileNetV2Small, self).__init__()

        # Initial channels
        input_channel = int(16 * width_mult)
        last_channel = int(640 * width_mult)

        # Inverted residual settings: [t, c, n, s]
        # t: expansion factor, c: output channels, n: repeat, s: stride
        inverted_residual_setting = [
            [1, 16, 1, 1],   # 28x28
            [6, 24, 2, 2],   # 14x14
            [6, 32, 3, 2],   # 7x7
            [6, 64, 4, 2],   # 4x4 (changed from stride=1 to fit 28x28)
            [6, 96, 3, 1],   # 4x4
            [6, 160, 3, 1],  # 4x4 (changed from stride=2)
            [6, 320, 1, 1],  # 4x4
        ]

        # Initial convolution
        self.features = [nn.Sequential(
            nn.Conv2d(1, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    self.InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        # Final convolution
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(last_channel, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 1, 28, 28]

        Returns:
            Output tensor [batch_size, 10]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class FashionMNISTCNN(nn.Module):
    """
    Configurable CNN for Fashion MNIST image classification.

    Defaults reproduce the previous architecture (Conv 32/64, kernel 3, FC 128).
    """

    def __init__(
        self,
        conv_channels=(32, 64),
        kernel_size=3,
        use_batchnorm=False,
        fc_units=128,
        dropout=0.5,
        padding="same",
    ):
        """
        Args:
            conv_channels: Tuple/list with channels per conv block (len 2 or 3 recommended)
            kernel_size: Kernel size for all conv layers
            use_batchnorm: If True, add BatchNorm2d after each conv
            fc_units: Hidden units in the first fully connected layer
            dropout: Dropout probability after FC1
            padding: 'same' (default) keeps spatial size before pooling; otherwise no padding
        """
        super(FashionMNISTCNN, self).__init__()

        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = 0

        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
            layers.append(conv)
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # Compute flatten size (28x28 input)
        spatial = 28
        for _ in conv_channels:
            spatial = spatial - (kernel_size - 1 - 2 * pad)
            spatial = spatial // 2
        flatten_dim = in_ch * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TextCNN(nn.Module):
    """
    TextCNN (Kim 2014) for AG News text classification

    Architecture:
    - Embedding layer (vocab_size -> 300d)
    - Multiple Conv1d with different filter sizes (3, 4, 5)
    - MaxPooling over time
    - Concatenate + FC -> 4 classes

    Parameters: ~2M
    Expected accuracy: 90-91%
    Training time: ~1-2 min/época
    """

    def __init__(self, vocab_size, embedding_dim=300, num_filters=100,
                 filter_sizes=[3, 4, 5], dropout=0.5):
        """
        Initialize TextCNN

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 300)
            num_filters: Number of filters per filter size (default: 100)
            filter_sizes: List of filter sizes (default: [3, 4, 5])
            dropout: Dropout probability (default: 0.5)
        """
        super(TextCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layers (one for each filter size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 4)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len] (token indices)

        Returns:
            Output tensor [batch_size, 4]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: [batch_size, num_filters, seq_len - filter_size + 1]
            conv_out = F.relu(conv(embedded))
            # MaxPool over time: [batch_size, num_filters, 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            # Squeeze: [batch_size, num_filters]
            pooled = pooled.squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate outputs: [batch_size, len(filter_sizes) * num_filters]
        concat = torch.cat(conv_outputs, dim=1)

        # Dropout + FC
        out = self.dropout(concat)
        out = self.fc(out)

        return out


class BiLSTMSimple(nn.Module):
    """
    Simple BiLSTM for AG News text classification

    Architecture:
    - Embedding(300d)
    - BiLSTM(128)
    - FC(256 -> 64)
    - FC(64 -> 4)

    Parameters: ~3M
    Expected accuracy: 91-92%
    Training time: ~3-4 min/época
    """

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, dropout=0.5):
        """
        Initialize BiLSTM Simple

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 300)
            hidden_dim: LSTM hidden dimension (default: 128)
            dropout: Dropout probability (default: 0.5)
        """
        super(BiLSTMSimple, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 4)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len] (token indices)

        Returns:
            Output tensor [batch_size, 4]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(x))

        # BiLSTM: [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim*2]
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate forward and backward hidden states
        # hidden: [2, batch_size, hidden_dim]
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim*2]

        # FC layers
        out = self.dropout(F.relu(self.fc1(hidden)))
        out = self.fc2(out)

        return out


class LSTMAttention(nn.Module):
    """
    LSTM + Attention for AG News text classification

    Architecture:
    - Embedding(300d)
    - BiLSTM(128)
    - Attention mechanism
    - FC(256 -> 4)

    Parameters: ~4M
    Expected accuracy: 92-93%
    Training time: ~4-5 min/época
    """

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, dropout=0.5):
        """
        Initialize LSTM + Attention

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 300)
            hidden_dim: LSTM hidden dimension (default: 128)
            dropout: Dropout probability (default: 0.5)
        """
        super(LSTMAttention, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, 4)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len] (token indices)

        Returns:
            Output tensor [batch_size, 4]
        """
        # Compute real sequence lengths (PAD=0); clamp to avoid zeros
        lengths = (x != 0).sum(dim=1).clamp(min=1)

        # Embedding
        embedded = self.dropout(self.embedding(x))

        # Pack LSTM to skip padded steps
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Build mask to ignore padding in attention
        mask = (torch.arange(lstm_out.size(1), device=lstm_out.device)
                .unsqueeze(0)
                .expand(lstm_out.size(0), -1))
        mask = (mask < lengths.unsqueeze(1)).unsqueeze(-1)  # [batch, seq, 1]

        attention_scores = self.attention(lstm_out)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)

        attended = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.dropout(attended)
        out = self.fc(out)

        return out


class AGNewsLSTM(nn.Module):
    """
    Basic LSTM for AG News text classification (original simple model)

    Architecture:
    - Embedding layer (vocab_size -> embedding_dim)
    - LSTM (embedding_dim -> hidden_dim)
    - FC: hidden_dim -> 4 classes
    """

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 num_layers=1, dropout=0.5, bidirectional=False):
        """
        Initialize LSTM

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 100)
            hidden_dim: LSTM hidden dimension (default: 128)
            num_layers: Number of LSTM layers (default: 1)
            dropout: Dropout probability (default: 0.5)
            bidirectional: Use bidirectional LSTM (default: False)
        """
        super(AGNewsLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, 4)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len] (token indices)

        Returns:
            Output tensor [batch_size, 4]
        """
        # Compute real sequence lengths (PAD=0); clamp to avoid zeros
        lengths = (x != 0).sum(dim=1).clamp(min=1)

        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(x))

        # Pack to ignore padding steps inside the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed)

        # hidden already corresponds to last valid step; stitch directions if needed
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        out = self.fc(self.dropout(hidden))

        return out


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT for AG News text classification

    Architecture:
    - DistilBERT (base-uncased)
    - Dropout
    - FC: 768 -> 4 classes

    Parameters: ~66M
    Expected accuracy: 94-95%
    Training time: ~15-20 min/época (batch 16, MPS-optimized)

    Note: Requires transformers library and uses pretrained weights
    """

    def __init__(self, dropout=0.1, pretrained_model="distilbert-base-uncased"):
        """
        Initialize DistilBERT Classifier

        Args:
            dropout: Dropout probability (default: 0.1)
            pretrained_model: Pretrained model name (default: distilbert-base-uncased)
        """
        super(DistilBERTClassifier, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not available. "
                "Install with: pip install transformers"
            )

        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)

        # Dropout and classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 4)  # DistilBERT hidden size is 768

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass

        Args:
            input_ids: Input tensor [batch_size, seq_len] (token indices)
            attention_mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            Output tensor [batch_size, 4]
        """
        # DistilBERT forward
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Get [CLS] token representation
        hidden_state = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # Dropout + classifier
        pooled = self.dropout(hidden_state)
        logits = self.classifier(pooled)

        return logits


class PyTorchClassifier:
    """
    Wrapper for PyTorch models to provide sklearn-like interface
    """

    def __init__(self, model, learning_rate=0.001, batch_size=64,
                 epochs=10, device=None, verbose=True, use_long_tensor=False,
                 log_to_mlflow=False):
        """
        Initialize wrapper

        Args:
            model: PyTorch model (nn.Module)
            learning_rate: Learning rate (default: 0.001)
            batch_size: Batch size (default: 64)
            epochs: Number of epochs (default: 10)
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None
            verbose: Print training progress (default: True)
            use_long_tensor: Use LongTensor for input (for LSTM with token indices) (default: False)
            log_to_mlflow: Log metrics per epoch to MLflow (default: False)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.use_long_tensor = use_long_tensor
        self.log_to_mlflow = log_to_mlflow

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
        }

    def fit(self, X, y):
        """
        Train the model

        Args:
            X: Training data (numpy array or tensor)
            y: Training labels (numpy array or tensor)

        Returns:
            self
        """
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader

        # Convert to tensors
        if isinstance(X, np.ndarray):
            if self.use_long_tensor:
                X = torch.LongTensor(X)
            else:
                X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        # Create DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

            # Epoch metrics
            avg_loss = epoch_loss / epoch_total
            avg_acc = epoch_correct / epoch_total

            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)

            # Log to MLflow if enabled
            if self.log_to_mlflow:
                try:
                    import mlflow
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", avg_acc, step=epoch)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not log to MLflow: {e}")

            if self.verbose:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')

        # Generate and log training curves if MLflow is enabled
        if self.log_to_mlflow and len(self.history['train_loss']) > 0:
            try:
                import mlflow
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Loss curve
                ax1.plot(self.history['train_loss'], 'b-', linewidth=2)
                ax1.set_title('Training Loss (Gradient Descent)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Epoch', fontsize=10)
                ax1.set_ylabel('Loss', fontsize=10)
                ax1.grid(True, alpha=0.3)

                # Accuracy curve
                ax2.plot(self.history['train_acc'], 'g-', linewidth=2)
                ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Epoch', fontsize=10)
                ax2.set_ylabel('Accuracy', fontsize=10)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                mlflow.log_figure(fig, "training_curves.png")
                plt.close(fig)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not log training curves to MLflow: {e}")

        return self

    def predict(self, X):
        """
        Predict labels

        Args:
            X: Test data (numpy array or tensor)

        Returns:
            Predicted labels (numpy array)
        """
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader

        # Convert to tensor
        if isinstance(X, np.ndarray):
            if self.use_long_tensor:
                X = torch.LongTensor(X)
            else:
                X = torch.FloatTensor(X)

        # Create DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Prediction loop
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Test data (numpy array or tensor)

        Returns:
            Class probabilities (numpy array)
        """
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader

        # Convert to tensor
        if isinstance(X, np.ndarray):
            if self.use_long_tensor:
                X = torch.LongTensor(X)
            else:
                X = torch.FloatTensor(X)

        # Create DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Prediction loop
        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)
