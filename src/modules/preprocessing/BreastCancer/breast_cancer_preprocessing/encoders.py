"""
Encoding functions for Breast Cancer dataset features and targets.

Feature types:
- age: Categorical (10-19, 20-29, ..., 90-99)
- menopause: Categorical (lt40, ge40, premeno)
- tumor-size: Categorical/Ordinal (0-4, 5-9, ..., 55-59)
- inv-nodes: Categorical/Ordinal (0-2, 3-5, ..., 36-39)
- node-caps: Binary (yes, no)
- deg-malig: Integer (1, 2, 3)
- breast: Binary (left, right)
- breast-quad: Categorical (left-up, left-low, right-up, right-low, central)
- irradiat: Binary (yes, no)

Target:
- Class: Binary (no-recurrence-events, recurrence-events)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# Ordinal mappings for ordinal features (order preserved, normalized to [0, 1])
ORDINAL_MAPPINGS = {
    'age': {
        '10-19': 0, '20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4,
        '60-69': 5, '70-79': 6, '80-89': 7, '90-99': 8
    },
    'tumor-size': {
        '0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4,
        '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9,
        '50-54': 10, '55-59': 11
    },
    'inv-nodes': {
        '0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5,
        '18-20': 6, '21-23': 7, '24-26': 8, '27-29': 9, '30-32': 10,
        '33-35': 11, '36-39': 12
    }
}

# Coordinate mapping for breast-quad (2D spatial representation)
BREAST_QUAD_COORDS = {
    'left_up': (-1, 1),
    'left-up': (-1, 1),  # alternative format
    'left_low': (-1, -1),
    'left-low': (-1, -1),  # alternative format
    'right_up': (1, 1),
    'right-up': (1, 1),  # alternative format
    'right_low': (1, -1),
    'right-low': (1, -1),  # alternative format
    'central': (0, 0),
}


def encode_target(y: pd.DataFrame, strategy="binary") -> torch.Tensor:
    """
    Encode target variable.

    Args:
        y: Target DataFrame with 'Class' column
        strategy: 'binary' (0/1) or 'label' (label encoding)

    Returns:
        torch.Tensor of shape (n_samples,) for binary, (n_samples, 2) for onehot
    """
    if strategy == "binary":
        # recurrence-events = 1, no-recurrence-events = 0
        encoded = (y['Class'] == 'recurrence-events').astype(int).values
        return torch.LongTensor(encoded)

    elif strategy == "label":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y['Class'])
        return torch.LongTensor(encoded)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def encode_breast_quad_coordinates(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode breast-quad as 2D coordinates (x, y) ∈ {-1, 0, 1}.

    Args:
        X: Features DataFrame with 'breast-quad' column

    Returns:
        DataFrame with 'breast-quad-x' and 'breast-quad-y' instead of 'breast-quad'
    """
    X_copy = X.copy()

    if 'breast-quad' in X_copy.columns:
        # Map to coordinates
        coords = X_copy['breast-quad'].map(BREAST_QUAD_COORDS)

        # Split into x and y columns
        X_copy['breast-quad-x'] = coords.apply(lambda c: c[0] if isinstance(c, tuple) else np.nan)
        X_copy['breast-quad-y'] = coords.apply(lambda c: c[1] if isinstance(c, tuple) else np.nan)

        # Drop original categorical column
        X_copy = X_copy.drop(columns=['breast-quad'])

    return X_copy


def encode_ordinal_continuous(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinal features (age, tumor-size, inv-nodes) as continuous [0, 1].

    Args:
        X: Features DataFrame

    Returns:
        DataFrame with ordinal features normalized to [0, 1]
    """
    X_copy = X.copy()

    # Fix UCI data corruption (ranges like '3-5' were converted to dates like '5-Mar')
    # Source data from UCI already has these corrupted values
    DATE_FIX_MAPPING = {
        # tumor-size
        '9-May': '5-9',
        'May-9': '5-9',
        '14-Oct': '10-14',
        'Oct-14': '10-14',
        # inv-nodes
        '5-Mar': '3-5',
        'Mar-5': '3-5',
        '8-Jun': '6-8',
        'Jun-8': '6-8',
        '11-Sep': '9-11',
        'Sep-11': '9-11',
        '14-Dec': '12-14',
        'Dec-14': '12-14',
    }

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in X_copy.columns:
            # Fix date-like values
            X_copy[col] = X_copy[col].replace(DATE_FIX_MAPPING)

            # Map categorical to ordinal integers
            X_copy[col] = X_copy[col].map(mapping)
            # Normalize to [0, 1]
            max_val = max(mapping.values())
            X_copy[col] = X_copy[col] / max_val

    return X_copy


def encode_features(
    X: pd.DataFrame,
    strategy="onehot",
    handle_missing="drop",
    encoder=None
) -> tuple[torch.Tensor, object]:
    """
    Encode all features in the dataset.

    Args:
        X: Features DataFrame
        strategy: 'onehot', 'label', 'ordinal', or 'mixed'
        handle_missing: 'drop' rows with missing values, 'most_frequent', or 'constant'
        encoder: Pre-fitted encoder (for test set). If None, fits a new encoder.

    Returns:
        (encoded_tensor, fitted_encoder) tuple
    """
    X_copy = X.copy()

    # Handle missing values (node-caps and breast-quad have missing)
    if handle_missing == "drop":
        X_copy = X_copy.dropna()
    elif handle_missing == "most_frequent":
        X_copy = X_copy.fillna(X_copy.mode().iloc[0])
    elif handle_missing == "constant":
        X_copy = X_copy.fillna("missing")

    if strategy == "mixed":
        # Mixed: ordinal features as continuous [0,1], breast-quad as (x,y), rest as onehot
        # Convert ordinal features to continuous
        X_copy = encode_ordinal_continuous(X_copy)

        # Convert breast-quad to coordinates
        X_copy = encode_breast_quad_coordinates(X_copy)

        # Separate continuous features (ordinal + breast-quad coords)
        ordinal_cols = list(ORDINAL_MAPPINGS.keys())
        coord_cols = ['breast-quad-x', 'breast-quad-y']
        continuous_cols = ordinal_cols + [c for c in coord_cols if c in X_copy.columns]
        continuous_features = X_copy[continuous_cols].values

        # Get remaining categorical features
        categorical_cols = [col for col in X_copy.columns if col not in continuous_cols]
        X_categorical = X_copy[categorical_cols]

        # OneHot encode categorical features
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_encoded = encoder.fit_transform(X_categorical)
        else:
            categorical_encoded = encoder.transform(X_categorical)

        # Concatenate: [continuous features, categorical features (onehot)]
        encoded = np.concatenate([continuous_features, categorical_encoded], axis=1)
        return torch.FloatTensor(encoded), encoder

    elif strategy == "onehot":
        # OneHot encoding for all categorical features
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(X_copy)
        else:
            encoded = encoder.transform(X_copy)
        return torch.FloatTensor(encoded), encoder

    elif strategy == "label":
        # Label encoding for each column
        if encoder is None:
            encoder = {}
            encoded_cols = []
            for col in X_copy.columns:
                le = LabelEncoder()
                encoded_cols.append(le.fit_transform(X_copy[col]))
                encoder[col] = le
            encoded = np.column_stack(encoded_cols)
        else:
            encoded_cols = []
            for col in X_copy.columns:
                encoded_cols.append(encoder[col].transform(X_copy[col]))
            encoded = np.column_stack(encoded_cols)
        return torch.FloatTensor(encoded), encoder

    elif strategy == "ordinal":
        # Ordinal encoding with natural ordering where applicable
        if encoder is None:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded = encoder.fit_transform(X_copy)
        else:
            encoded = encoder.transform(X_copy)
        return torch.FloatTensor(encoded), encoder

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def encode_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_strategy="onehot",
    target_strategy="binary",
    handle_missing="drop",
    encoder=None
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """
    Encode both features and targets.

    Args:
        X: Features DataFrame
        y: Target DataFrame
        feature_strategy: Encoding strategy for features
        target_strategy: Encoding strategy for target
        handle_missing: How to handle missing values
        encoder: Pre-fitted encoder (for test set). If None, fits a new encoder.

    Returns:
        (X_encoded, y_encoded, fitted_encoder) tuple
    """
    # Handle missing values - align indices
    if handle_missing == "drop":
        # Drop rows with missing values and align X and y
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    X_encoded, fitted_encoder = encode_features(
        X, strategy=feature_strategy, handle_missing=handle_missing, encoder=encoder
    )
    y_encoded = encode_target(y, strategy=target_strategy)

    return X_encoded, y_encoded, fitted_encoder
