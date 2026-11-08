"""Read Fashion-MNIST IDX files and prepare train, validation, and test arrays."""

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

RAW_DIR = (Path(__file__).resolve().parent / ".." / ".." / "FashionMNIST" / "raw").resolve()

RawImages = NDArray[np.uint8]
Features = NDArray[np.float32]
Labels = NDArray[np.int64]


def load_images(filepath: str | Path) -> RawImages:
    """Read an uncompressed IDX3 file into a uint8 array of shape (N, H, W)."""
    with Path(filepath).open("rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"Incomplete image header: {filepath}")

        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid image magic number {magic}: {filepath}")
        if count == 0 or rows == 0 or cols == 0:
            raise ValueError(f"Invalid image count or dimensions: {filepath}")

        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    expected_size = count * rows * cols
    if pixels.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} pixels, found {pixels.size}: {filepath}"
        )

    return pixels.reshape(count, rows, cols)


def load_labels(filepath: str | Path) -> Labels:
    """Read an uncompressed IDX1 file into an int64 vector of class IDs."""
    with Path(filepath).open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"Incomplete label header: {filepath}")

        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid label magic number {magic}: {filepath}")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.size != count:
        raise ValueError(
            f"Expected {count} labels, found {labels.size}: {filepath}"
        )
    if np.any(labels > 9):
        raise ValueError(f"Fashion-MNIST labels must be between 0 and 9: {filepath}")

    return labels.astype(np.int64)


def load_fashion_mnist(
    raw_dir: str | Path | None = None,
) -> tuple[RawImages, Labels, RawImages, Labels]:
    """Load and pair the original Fashion-MNIST training and test arrays."""
    directory = Path(RAW_DIR if raw_dir is None else raw_dir)
    X_train = load_images(directory / "train-images-idx3-ubyte")
    y_train = load_labels(directory / "train-labels-idx1-ubyte")
    X_test = load_images(directory / "t10k-images-idx3-ubyte")
    y_test = load_labels(directory / "t10k-labels-idx1-ubyte")

    if len(X_train) != len(y_train):
        raise ValueError("Training image and label counts do not match.")
    if len(X_test) != len(y_test):
        raise ValueError("Test image and label counts do not match.")

    return X_train, y_train, X_test, y_test


def normalize_data(X: RawImages) -> Features:
    """Convert pixel intensities to float32 and scale them to [0, 1]."""
    scaled = X.astype(np.float32)
    scaled /= np.float32(255.0)
    return scaled


def flatten_data(X: Features) -> Features:
    """Flatten the dimensions after the sample axis."""
    return X.reshape(X.shape[0], -1)


def preprocess_images(images: RawImages, flatten: bool = True) -> Features:
    """Scale 28×28 images and optionally flatten each image to 784 features."""
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected Fashion-MNIST images with shape (N, 28, 28).")

    scaled = normalize_data(images)
    return flatten_data(scaled) if flatten else scaled


def split_validation(
    X_train: Features,
    y_train: Labels,
    val_size: int | float = 0.2,
    random_state: int = 42,
) -> tuple[Features, Features, Labels, Labels]:
    """Split the original training data while preserving class proportions."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train, X_val, y_train, y_val


def preprocess_fashion_mnist(
    flatten: bool = True,
    val_size: int | float = 10_000,
    *,
    raw_dir: str | Path | None = None,
) -> tuple[Features, Features, Features, Labels, Labels, Labels]:
    """Return (X_train, X_val, X_test, y_train, y_val, y_test).

    The default reserves 10,000 training samples for validation. Pass a
    fraction such as 0.2 to select validation size proportionally.
    """
    X_train_full, y_train_full, X_test_raw, y_test = load_fashion_mnist(raw_dir)
    X_train_full_scaled = preprocess_images(X_train_full, flatten=flatten)
    X_test = preprocess_images(X_test_raw, flatten=flatten)
    X_train, X_val, y_train, y_val = split_validation(
        X_train_full_scaled, y_train_full, val_size=val_size
    )

    print("===== DATASET FASHIONMNIST INFORMATION =====")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test
