"""
Tokenized Data loader for AG_NEWS (for LSTM/RNN models)

This loader returns token sequences instead of TF-IDF features,
making it suitable for neural network models like LSTM.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import urllib.request
import shutil
from collections import Counter
import re

from src.config import DATA_DIR, RANDOM_SEED


class AGNewsTokenizedLoader:
    """
    Tokenized loader for AG_NEWS dataset
    Returns padded token sequences suitable for LSTM/RNN models
    """

    def __init__(self, max_vocab_size=10000, max_seq_length=200):
        """
        Args:
            max_vocab_size: Maximum vocabulary size (default: 10000)
            max_seq_length: Maximum sequence length (default: 200)
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.random_seed = RANDOM_SEED

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

        # Vocabulary mappings
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_size = 2  # Start with PAD and UNK

        # Diretório para cache
        self.data_dir = DATA_DIR / "ag_news"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Carregar datasets
        print("Loading AG_NEWS dataset...")
        self.train_data = self._load_ag_news_csv('train')
        self.test_data = self._load_ag_news_csv('test')

        print(f"  Train samples: {len(self.train_data)}")
        print(f"  Test samples: {len(self.test_data)}")

    def _download_if_needed(self, split):
        """Download AG_NEWS CSV files if not present"""
        urls = {
            'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
            'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
        }

        filepath = self.data_dir / f"{split}.csv"

        if not filepath.exists():
            print(f"  Downloading {split}.csv...")
            try:
                with urllib.request.urlopen(urls[split]) as response, open(filepath, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                print(f"  Error downloading from GitHub, trying alternative source...")
                # Alternative: use torchtext cache if available
                alt_path = self.data_dir / '.data' / 'ag_news_csv' / f'{split}.csv'
                if alt_path.exists():
                    shutil.copy(alt_path, filepath)
                else:
                    raise Exception(f"Could not download AG_NEWS {split} data: {e}")

        return filepath

    def _load_ag_news_csv(self, split):
        """Load AG_NEWS from CSV file

        CSV format: class,title,description
        Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech (we convert to 0-indexed)
        """
        filepath = self._download_if_needed(split)

        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    label = int(row[0]) - 1  # Convert to 0-indexed
                    title = row[1]
                    description = row[2]
                    text = f"{title}. {description}"  # Combine title and description
                    data.append((label, text))

        return data

    def _tokenize(self, text):
        """
        Simple tokenization: lowercase, split on non-alphanumeric

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _build_vocabulary(self, texts):
        """
        Build vocabulary from texts

        Args:
            texts: List of text strings
        """
        print(f"\nBuilding vocabulary (max_vocab_size={self.max_vocab_size})...")

        # Count all words
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        print(f"  Total unique words: {len(word_counts)}")

        # Keep most common words (excluding PAD and UNK)
        most_common = word_counts.most_common(self.max_vocab_size - 2)

        # Build word2idx and idx2word
        for idx, (word, count) in enumerate(most_common, start=2):  # Start at 2 (after PAD, UNK)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Most common words: {most_common[:10]}")

    def _text_to_sequence(self, text):
        """
        Convert text to sequence of token indices

        Args:
            text: Input text string

        Returns:
            List of token indices
        """
        tokens = self._tokenize(text)
        sequence = [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]
        return sequence

    def _pad_sequences(self, sequences):
        """
        Pad sequences to max_seq_length

        Args:
            sequences: List of token sequences

        Returns:
            Numpy array of padded sequences
        """
        padded = np.zeros((len(sequences), self.max_seq_length), dtype=np.int32)

        for i, seq in enumerate(sequences):
            # Truncate if too long
            seq = seq[:self.max_seq_length]
            # Copy sequence
            padded[i, :len(seq)] = seq

        return padded

    def get_numpy_arrays(self, val_split=0.2):
        """
        Return data as numpy arrays with tokenized sequences

        Args:
            val_split: Proportion of training set for validation

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
            All X arrays are shape (n_samples, max_seq_length) with token indices
        """
        # Prepare data
        train_labels = [label for label, text in self.train_data]
        train_texts = [text for label, text in self.train_data]
        test_labels = [label for label, text in self.test_data]
        test_texts = [text for label, text in self.test_data]

        # Split train/validation
        train_texts_split, val_texts, y_train, y_val = train_test_split(
            train_texts, train_labels,
            test_size=val_split,
            random_state=self.random_seed,
            stratify=train_labels
        )

        # Build vocabulary on training set only
        self._build_vocabulary(train_texts_split)

        # Convert texts to sequences
        print(f"\nTokenizing texts (max_seq_length={self.max_seq_length})...")
        train_sequences = [self._text_to_sequence(text) for text in train_texts_split]
        val_sequences = [self._text_to_sequence(text) for text in val_texts]
        test_sequences = [self._text_to_sequence(text) for text in test_texts]

        # Compute sequence length statistics
        train_lens = [len(seq) for seq in train_sequences]
        print(f"  Train sequence lengths - min: {min(train_lens)}, max: {max(train_lens)}, mean: {np.mean(train_lens):.1f}")

        # Pad sequences
        X_train = self._pad_sequences(train_sequences)
        X_val = self._pad_sequences(val_sequences)
        X_test = self._pad_sequences(test_sequences)

        # Convert labels to numpy
        y_train = np.array(y_train, dtype=np.int64)
        y_val = np.array(y_val, dtype=np.int64)
        y_test = np.array(test_labels, dtype=np.int64)

        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  Vocabulary size: {self.vocab_size}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_class_names(self):
        """Return class names"""
        return ["World", "Sports", "Business", "Sci/Tech"]

    def get_dataset_info(self):
        """Return dataset information"""
        info = {
            "train_samples": len(self.train_data),
            "test_samples": len(self.test_data),
            "num_classes": 4,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "max_vocab_size": self.max_vocab_size,
            "tokenization": "word-level",
        }
        return info


if __name__ == "__main__":
    # Test the tokenized data loader
    print("Loading AG_NEWS with tokenization...")
    loader = AGNewsTokenizedLoader(max_vocab_size=10000, max_seq_length=200)

    # Dataset info
    info = loader.get_dataset_info()
    print("\nDataset information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Load as numpy arrays
    print("\nLoading data as numpy arrays...")
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  X_val: {X_val.shape}, dtype: {X_val.dtype}")
    print(f"  y_val: {y_val.shape}, dtype: {y_val.dtype}")
    print(f"  X_test: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"  y_test: {y_test.shape}, dtype: {y_test.dtype}")

    # Data statistics
    print(f"\nData statistics:")
    print(f"  y_train min/max: {y_train.min()}/{y_train.max()}, unique: {np.unique(y_train)}")
    print(f"  X_train min/max: {X_train.min()}/{X_train.max()}")
    print(f"  Train class distribution: {np.bincount(y_train)}")
    print(f"  Val class distribution: {np.bincount(y_val)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")

    # Check padding
    print(f"\nPadding check:")
    print(f"  % of padding in X_train: {(X_train == 0).sum() / X_train.size * 100:.2f}%")
    print(f"  Sample sequence (first 20 tokens): {X_train[0][:20]}")

    print("\n✓ AG_NEWS tokenized data loader working correctly!")
