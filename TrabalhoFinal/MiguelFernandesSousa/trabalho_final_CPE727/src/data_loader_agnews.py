"""
Data loader para AG_NEWS

O dataset AG_NEWS é automaticamente baixado pela biblioteca torchtext.
Contém notícias categorizadas em 4 classes: World, Sports, Business, Sci/Tech.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import urllib.request
import shutil

from src.config import DATA_DIR, RANDOM_SEED


class AGNewsLoader:
    """
    Classe para carregar e pré-processar o dataset AG_NEWS
    """

    def __init__(self, max_features=10000, max_df=0.5, min_df=5):
        """
        Args:
            max_features: Número máximo de features TF-IDF
            max_df: Ignora termos que aparecem em mais de max_df dos documentos
            min_df: Ignora termos que aparecem em menos de min_df documentos
        """
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.random_seed = RANDOM_SEED
        self.vectorizer = None

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

    def _prepare_data(self, data):
        """
        Extrai textos e labels do dataset

        Args:
            data: Lista de tuplas (label, text)

        Returns:
            texts, labels (listas)
        """
        labels = [label for label, text in data]
        texts = [text for label, text in data]
        return texts, labels

    def get_numpy_arrays(self, val_split=0.2):
        """
        Retorna dados como arrays numpy com TF-IDF (útil para scikit-learn)

        Args:
            val_split: Proporção do conjunto de treino para validação

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Preparar dados
        train_texts, y_train_full = self._prepare_data(self.train_data)
        test_texts, y_test = self._prepare_data(self.test_data)

        # Converter para numpy arrays
        y_train_full = np.array(y_train_full)
        y_test = np.array(y_test)

        # Split treino/validação
        train_texts_split, val_texts, y_train, y_val = train_test_split(
            train_texts, y_train_full,
            test_size=val_split,
            random_state=self.random_seed,
            stratify=y_train_full
        )

        # Converter para numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        # Criar e fit TF-IDF vectorizer apenas no conjunto de treino
        print(f"\nVectorizing text with TF-IDF (max_features={self.max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            sublinear_tf=True,  # Use log scaling
            stop_words='english'
        )

        # Fit apenas no treino
        X_train = self.vectorizer.fit_transform(train_texts_split)

        # Transform val e test
        X_val = self.vectorizer.transform(val_texts)
        X_test = self.vectorizer.transform(test_texts)

        # Converter para dense arrays (necessário para alguns modelos)
        X_train = X_train.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  X_test shape: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_class_names(self):
        """Retorna os nomes das classes"""
        return ["World", "Sports", "Business", "Sci/Tech"]

    def get_dataset_info(self):
        """Retorna informações sobre o dataset"""
        info = {
            "train_samples": len(self.train_data),
            "test_samples": len(self.test_data),
            "num_classes": 4,
            "max_features": self.max_features,
            "vectorizer": "TF-IDF",
        }
        return info


if __name__ == "__main__":
    # Teste do data loader
    print("Carregando AG_NEWS...")
    loader = AGNewsLoader(max_features=10000)

    # Informações do dataset
    info = loader.get_dataset_info()
    print("\nInformações do dataset:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Testar carregamento como numpy
    print("\nCarregando dados como numpy arrays...")
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"  X_val: {X_val.shape}, dtype: {X_val.dtype}")
    print(f"  y_val: {y_val.shape}, dtype: {y_val.dtype}")
    print(f"  X_test: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"  y_test: {y_test.shape}, dtype: {y_test.dtype}")

    # Estatísticas dos dados
    print(f"\nData statistics:")
    print(f"  y_train min/max: {y_train.min()}/{y_train.max()}, unique: {np.unique(y_train)}")
    print(f"  y_val min/max: {y_val.min()}/{y_val.max()}, unique: {np.unique(y_val)}")
    print(f"  y_test min/max: {y_test.min()}/{y_test.max()}, unique: {np.unique(y_test)}")
    if y_train.min() >= 0:
        print(f"  Train class distribution: {np.bincount(y_train)}")
        print(f"  Val class distribution: {np.bincount(y_val)}")
        print(f"  Test class distribution: {np.bincount(y_test)}")
    print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  X_train sparsity: {(X_train == 0).sum() / X_train.size * 100:.2f}% zeros")

    print("\n✓ AG_NEWS data loader funcionando corretamente!")
