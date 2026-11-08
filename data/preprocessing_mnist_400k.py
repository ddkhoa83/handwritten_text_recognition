"""Preprocess expanded MNIST with the same module-level API as preprocessing.py.

Place this file in data/ and extract the dataset as:
    data/MNIST/raw/<four uncompressed IDX files>
    data/MNIST/metadata/train_rotations.csv
    data/MNIST/metadata/t10k_rotations.csv

Dataset: 360,000 training images and 40,000 test images.
Digits 6 and 9 are rotated within [-15,15] degrees; others within [0,360).
Importing this module loads and normalizes data immediately, like the sample.
Dependencies: numpy and scikit-learn. Allow roughly 2 GB of free RAM.

Validation contains 10,000 ORIGINAL sources and all their rotated variants,
so its row count is about 60,000 rather than exactly 10,000.
No extra augmentation or relabeling is performed here.
"""

from pathlib import Path
import csv
import struct

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

# Resolve data relative to this Python file, independent of notebook location.
data_dir: Path = Path(__file__).resolve().parent / "MNIST_expanded_400k_360k_train_40k_test_pm15/MNIST" / "raw"
metadata_dir: Path = data_dir.parent / "metadata"

TRAIN_SOURCE_COUNT: int = 60_000
TEST_SOURCE_COUNT: int = 10_000
N_FEATURES: int = 784
VALIDATION_SOURCE_COUNT: int = 10_000
RANDOM_STATE: int = 42

def load_mnist_images(file_path: str | Path) -> NDArray[np.uint8]:
    """Return read-only uint8 (N,28,28) IDX images using memory mapping.

    Parameters
    ----------
    file_path : str | Path
        Location of one uncompressed IDX image file.

    Returns
    -------
    NDArray[np.uint8]
        Image tensor with values 0..255.

    Raises
    ------
    ValueError
        If the IDX header, image dimensions, or file length is invalid.
    """
    path = Path(file_path)
    with path.open("rb") as file:
        header = file.read(16)
    if len(header) != 16:
        raise ValueError(f"Incomplete image IDX header: {path}")
    magic, count, height, width = struct.unpack(">IIII", header)
    if magic != 2051 or count == 0 or (height, width) != (28, 28):
        raise ValueError(f"Invalid MNIST image IDX shape/header: {path}")
    expected_size = 16 + count * height * width
    if path.stat().st_size != expected_size:
        raise ValueError(f"Image IDX file has wrong size: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8,
                     offset=16, shape=(count, height, width))


def load_mnist_labels(file_path: str | Path) -> NDArray[np.int64]:
    """Read IDX digit labels as a one-dimensional int64 array (0..9).

    Parameters
    ----------
    file_path : str | Path
        Location of one uncompressed IDX label file.

    Returns
    -------
    NDArray[np.int64]
        Label array of shape (N,).
    """
    path = Path(file_path)
    with path.open("rb") as file:
        header = file.read(8)
    if len(header) != 8:
        raise ValueError(f"Incomplete label IDX header: {path}")
    magic, count = struct.unpack(">II", header)
    if magic != 2049 or count == 0:
        raise ValueError(f"Invalid MNIST label IDX header: {path}")
    if path.stat().st_size != 8 + count:
        raise ValueError(f"Label IDX file has wrong size: {path}")
    labels_uint8 = np.memmap(path, mode="r", dtype=np.uint8,
                             offset=8, shape=(count,))
    if np.any(labels_uint8 > 9):
        raise ValueError(f"Labels outside the range 0..9: {path}")
    return labels_uint8.astype(np.int64)


def preprocess_images(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Flatten images to (N,784) and scale uint8 pixels to float32 [0,1].

    Parameters
    ----------
    images : NDArray[np.uint8]
        Input tensor of shape (N,28,28).

    Returns
    -------
    NDArray[np.float32]
        Flattened feature matrix of shape (N,784).
    """
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Expected image shape (N, 28, 28).")
    if images.dtype != np.uint8:
        raise TypeError(f"Expected uint8 pixels, got {images.dtype}.")
    scaled = np.asarray(images).reshape(len(images), N_FEATURES).astype(np.float32)
    scaled /= np.float32(255.0)
    return scaled


def load_source_ids(
    manifest_path: str | Path,
    labels: NDArray[np.int64],
    source_count: int,
    verify_manifest: bool = True,
) -> NDArray[np.int64]:
    """Read explicit source mapping; reject missing, duplicate or invalid rows.

    labels includes originals first, followed by augmented images.
    Mapping and labels are ALWAYS validated. verify_manifest additionally
    checks the expected angle policy. Metadata is never optional because
    row modulo source_count is incorrect for this dataset.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; extract the complete metadata/ folder.")
    result = np.full(len(labels), -1, dtype=np.int64)
    result[:source_count] = np.arange(source_count)
    required = {"output_index", "source_index", "label", "angle_degrees"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid CSV columns: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                output = int(row["output_index"])
                source = int(row["source_index"])
                label = int(row["label"])
                angle = float(row["angle_degrees"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid metadata at {path}:{line}") from error
            if not source_count <= output < len(labels):
                raise ValueError(f"Invalid output index at {path}:{line}")
            if not 0 <= source < source_count or result[output] != -1:
                raise ValueError(f"Invalid source or duplicate output at {path}:{line}")
            if label != labels[output] or label != labels[source]:
                raise ValueError(f"Source/output label mismatch at {path}:{line}")
            if verify_manifest:
                valid_angle = np.isfinite(angle) and (
                    -15 <= angle <= 15 if label in (6, 9) else 0 <= angle < 360
                )
                if not valid_angle:
                    raise ValueError(f"Angle outside dataset policy at {path}:{line}")
            result[output] = source
    if np.any(result < 0):
        raise ValueError(f"Metadata does not cover every augmented image: {path}")
    return result


def _normalize_selected(
    images: NDArray[np.uint8],
    indices: NDArray[np.int64],
    output_file: Path | None,
    batch_size: int,
) -> NDArray[np.float32]:
    """Normalize selected IDX rows in chunks, optionally to a disk memmap."""
    count = len(indices)
    if output_file is None:
        output: NDArray[np.float32] = np.empty((count, N_FEATURES), dtype=np.float32)
    else:
        output = np.memmap(output_file, dtype=np.float32, mode="w+",
                           shape=(count, N_FEATURES))

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        selected_images = images[indices[start:stop]]
        output[start:stop] = preprocess_images(selected_images)

    if isinstance(output, np.memmap):
        output.flush()
    return output


# Read uint8 pixels without loading every image into memory at once.
train_images: NDArray[np.uint8] = load_mnist_images(data_dir / "train-images-idx3-ubyte")
test_images: NDArray[np.uint8] = load_mnist_images(data_dir / "t10k-images-idx3-ubyte")

# Class labels remain integer values; they are never divided by 255.
train_labels: NDArray[np.int64] = load_mnist_labels(data_dir / "train-labels-idx1-ubyte")
test_labels: NDArray[np.int64] = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte")

# Refuse a different edition instead of silently applying the wrong source map.
if len(train_images) != 360_000 or len(train_labels) != 360_000:
    raise ValueError("Expected 360,000 training images and labels.")
if len(test_images) != 40_000 or len(test_labels) != 40_000:
    raise ValueError("Expected 40,000 test images and labels.")

# Augmented rows are NOT aligned blocks in this edition.
# CSV output_index identifies a row; source_index identifies its original image.
train_source_ids: NDArray[np.int64] = load_source_ids(
    metadata_dir / "train_rotations.csv", train_labels, TRAIN_SOURCE_COUNT,
)
test_source_ids: NDArray[np.int64] = load_source_ids(
    metadata_dir / "t10k_rotations.csv", test_labels, TEST_SOURCE_COUNT,
)

# Split original source IDs first, stratifying by the ORIGINAL digit labels.
# This keeps every original image together with all its rotated copies.
source_indices: NDArray[np.int64] = np.arange(TRAIN_SOURCE_COUNT, dtype=np.int64)
train_sources, val_sources = train_test_split(
    source_indices,
    test_size=VALIDATION_SOURCE_COUNT,
    random_state=RANDOM_STATE,
    stratify=train_labels[:TRAIN_SOURCE_COUNT],
)

# Look up each row's membership using its source ID, not its row position.
validation_source_mask = np.zeros(TRAIN_SOURCE_COUNT, dtype=bool)
validation_source_mask[val_sources] = True
validation_row_mask = validation_source_mask[train_source_ids]
train_indices: NDArray[np.int64] = np.flatnonzero(~validation_row_mask)
val_indices: NDArray[np.int64] = np.flatnonzero(validation_row_mask)

# Mix original and rotated images reproducibly for downstream training.
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(train_indices)
rng.shuffle(val_indices)
if np.intersect1d(train_source_ids[train_indices], train_source_ids[val_indices]).size:
    raise RuntimeError("Training and validation share original image sources.")

# Normalize in small chunks to reduce temporary memory allocations.
# Every feature matrix has 784 columns and float32 values in [0,1].
X_train_scaled: NDArray[np.float32] = _normalize_selected(
    train_images, train_indices, output_file=None, batch_size=2048,
)
X_val_scaled: NDArray[np.float32] = _normalize_selected(
    train_images, val_indices, output_file=None, batch_size=2048,
)
X_test_scaled: NDArray[np.float32] = _normalize_selected(
    test_images, np.arange(len(test_images), dtype=np.int64),
    output_file=None, batch_size=2048,
)

# Select labels with the SAME indices used for the feature rows.
y_train: NDArray[np.int64] = train_labels[train_indices]
y_val: NDArray[np.int64] = train_labels[val_indices]
y_test: NDArray[np.int64] = test_labels

# Preserve the full training labels used in the reference file.
# A full normalized X_train_full is intentionally not allocated as a duplicate.
y_train_full: NDArray[np.int64] = train_labels

if __name__ == "__main__":
    for name in ("X_train_scaled", "X_val_scaled", "X_test_scaled",
                 "y_train", "y_val", "y_test"):
        array = globals()[name]
        print(f"{name:16s}: shape={array.shape}, dtype={array.dtype}")
    print("Verified: training and validation source IDs are disjoint.")
