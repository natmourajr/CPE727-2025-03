import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from tokenizers import Tokenizer
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model


def get_fasttext_dimension(model_path: str) -> int:
    """
    Get embedding dimension from FastText model without fully loading vectors.

    Parameters
    ----------
    model_path : str
        Path to FastText .bin file

    Returns
    -------
    int
        Embedding dimension of the FastText model

    Raises
    ------
    FileNotFoundError
        If model_path doesn't exist
    ValueError
        If file is not a valid FastText model
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FastText model file not found: {model_path}")

    try:
        # Load model to get dimension
        ft_model = load_facebook_model(str(path))
        dimension = ft_model.wv.vector_size
        del ft_model  # Free memory
        return dimension
    except Exception as e:
        raise ValueError(f"Failed to load FastText model from {model_path}: {e}")


def load_fasttext_model(model_path: str) -> KeyedVectors:
    """
    Load FastText binary model (.bin) using gensim.

    Parameters
    ----------
    model_path : str
        Path to FastText .bin file

    Returns
    -------
    KeyedVectors
        Loaded FastText KeyedVectors (supports subword queries)

    Raises
    ------
    FileNotFoundError
        If model_path doesn't exist
    ValueError
        If file is not a valid FastText model
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FastText model file not found: {model_path}")

    try:
        # Load full FastText model
        ft_model = load_facebook_model(str(path))
        # Return just the KeyedVectors (word vectors with subword support)
        return ft_model.wv
    except Exception as e:
        raise ValueError(f"Failed to load FastText model from {model_path}: {e}")


def create_embedding_matrix(
    tokenizer: Tokenizer,
    fasttext_model: KeyedVectors,
    embedding_dim: int,
    padding_idx: int = 0,
) -> torch.Tensor:
    """
    Create embedding matrix from FastText model aligned with tokenizer vocabulary.

    FastText's subword approach automatically handles OOV words by computing
    vectors from character n-grams.

    Process:
    1. Initialize matrix with zeros (vocab_size x embedding_dim)
    2. For each token in vocabulary:
       - Get FastText vector (uses subword if OOV)
       - Place in corresponding row
    3. Keep padding vector as zeros
    4. Handle special tokens (<UNK>, <EOS>) appropriately

    Parameters
    ----------
    tokenizer : Tokenizer
        Trained tokenizer with vocabulary
    fasttext_model : KeyedVectors
        Loaded FastText KeyedVectors
    embedding_dim : int
        Expected embedding dimension (must match FastText model)
    padding_idx : int, default=0
        Index of padding token (will remain zeros)

    Returns
    -------
    torch.Tensor
        Embedding matrix of shape (vocab_size, embedding_dim)

    Raises
    ------
    ValueError
        If FastText dimension doesn't match embedding_dim
    """
    # Validate dimension
    if fasttext_model.vector_size != embedding_dim:
        raise ValueError(
            f"FastText model has dimension {fasttext_model.vector_size}, "
            f"but expected {embedding_dim}"
        )

    vocab_size = tokenizer.get_vocab_size()
    vocab = tokenizer.get_vocab()

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Track statistics
    found_count = 0
    oov_count = 0

    # Fill embedding matrix
    for token, idx in vocab.items():
        # Skip padding token (keep as zeros)
        if idx == padding_idx:
            continue

        try:
            # FastText automatically handles OOV via subword vectors
            # This works for both in-vocabulary and out-of-vocabulary words
            vector = fasttext_model[token]
            embedding_matrix[idx] = vector
            found_count += 1
        except KeyError:
            # This should rarely happen with FastText since it uses subwords
            # If it does, initialize with small random values
            embedding_matrix[idx] = np.random.uniform(
                -0.01, 0.01, size=embedding_dim
            ).astype(np.float32)
            oov_count += 1

    print(f"Embedding matrix created: {vocab_size} tokens, {embedding_dim} dimensions")
    print(f"  - Found/estimated: {found_count} tokens")
    if oov_count > 0:
        print(f"  - Randomly initialized: {oov_count} tokens (rare with FastText)")
    print(f"  - Padding token (zeros): 1 token")

    return torch.from_numpy(embedding_matrix)


def initialize_embeddings_from_fasttext(
    embedding_layer: nn.Embedding,
    embedding_matrix: torch.Tensor,
    freeze: bool = False
) -> nn.Embedding:
    """
    Initialize PyTorch embedding layer with pre-trained weights.

    Parameters
    ----------
    embedding_layer : nn.Embedding
        PyTorch embedding layer to initialize
    embedding_matrix : torch.Tensor
        Pre-trained embedding matrix (vocab_size x embedding_dim)
    freeze : bool, default=False
        If True, freeze embeddings (not trainable)
        If False, allow fine-tuning

    Returns
    -------
    nn.Embedding
        Initialized embedding layer
    """
    # Copy weights
    embedding_layer.weight.data.copy_(embedding_matrix)

    # Freeze if requested
    if freeze:
        embedding_layer.weight.requires_grad = False

    return embedding_layer
