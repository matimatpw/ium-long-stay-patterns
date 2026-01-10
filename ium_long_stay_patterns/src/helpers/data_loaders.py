"""Helpers for creating PyTorch DataLoaders from tensors."""

from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_loaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create and return (train_loader, val_loader, test_loader).

    Args:
        X_train, y_train: tensors for training set
        X_val, y_val: tensors for validation set
        X_test, y_test: tensors for test set
        batch_size: batch size for all loaders (default: 32)
        shuffle_train: whether to shuffle the training loader

    Returns:
        A tuple of three DataLoader objects: (train_loader, val_loader, test_loader)
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_and_create_loaders(X, y, batch_size=32):
    X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_raw)
    X_val_np = scaler.transform(X_val_raw)
    X_test_np = scaler.transform(X_test_raw)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    y_train = torch.tensor(y_train_raw.values, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val_raw.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test_raw.values, dtype=torch.float32).view(-1, 1)

    train_loader, val_loader, test_loader = create_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )

    print(f"Train set size: {X_train.shape[0]} ({y_train.sum().item()} positive)")
    print(f"Validation set size: {X_val.shape[0]} ({y_val.sum().item()} positive)")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Total positive samples in train set: {y_train.sum().item()}")

    return train_loader, val_loader, test_loader, scaler
