
import os
import gzip
import struct
import urllib.request
import numpy as np

from sklearn.model_selection import train_test_split


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "FashionMNIST", "raw")
)



# ==========================================
# 3. ĐỌC DỮ LIỆU IDX
# ==========================================


def load_images(filepath):
    """
    Đọc file ảnh IDX3.
    Trả về mảng numpy có dạng (N, 28, 28).
    """
    # Thay gzip.open thành open
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("File anh IDX khong hop le!")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def load_labels(filepath):
    """
    Đọc file nhãn IDX1.
    Trả về mảng numpy có dạng (N,).
    """
    # Thay gzip.open thành open
    with open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("File nhan IDX khong hop le!")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# ==========================================
# 4. LOAD TOÀN BỘ DATASET
# ==========================================

def load_fashion_mnist():
    # Xóa toàn bộ đuôi .gz trong tên file
    X_train = load_images(
        os.path.join(RAW_DIR, "train-images-idx3-ubyte")
    )

    y_train = load_labels(
        os.path.join(RAW_DIR, "train-labels-idx1-ubyte")
    )

    X_test = load_images(
        os.path.join(RAW_DIR, "t10k-images-idx3-ubyte")
    )

    y_test = load_labels(
        os.path.join(RAW_DIR, "t10k-labels-idx1-ubyte")
    )

    return X_train, y_train, X_test, y_test

# ==========================================
# 5. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================

def normalize_data(X):
    """
    Chuẩn hóa pixel từ [0, 255] về [0, 1].
    """
    X = X.astype(np.float32)

    X = X / 255.0

    return X


def flatten_data(X):
    """
    Chuyển ảnh (28, 28) thành vector (784).
    """
    return X.reshape(X.shape[0], -1)


def split_validation(X_train, y_train,
                     val_size=0.2, random_state=42):
    """
    Chia tập train thành train và validation.
    """
    return train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )


# ==========================================
# 6. PIPELINE TIỀN XỬ LÝ
# ==========================================

def preprocess_fashion_mnist(flatten=True,
                             val_size=0.2):
    """
    Pipeline tiền xử lý Fashion MNIST.

    flatten=True:
        Ảnh được chuyển thành vector 784 chiều.

    flatten=False:
        Giữ nguyên ảnh 28x28.
    """

    # Load raw data
    X_train_fs, y_train_fs, X_test_fs, y_test_fs = load_fashion_mnist()

    # Normalize
    X_train_fs = normalize_data(X_train_fs)
    X_test_fs = normalize_data(X_test_fs)

    # Flatten nếu cần
    if flatten:
        X_train_fs = flatten_data(X_train_fs)
        X_test_fs    = flatten_data(X_test_fs)

    # Split train / validation
    X_train_fs, X_val_fs, y_train_fs, y_val_fs = split_validation(
        X_train_fs, y_train_fs,
        val_size=val_size
    )

    print("===== DATASET FASHIONMNIST INFORMATION =====")
    print("Train:", X_train_fs.shape, y_train_fs.shape)
    print("Validation:", X_val_fs.shape, y_val_fs.shape)
    print("Test:", X_test_fs.shape, y_test_fs.shape)

    return X_train_fs, X_val_fs, X_test_fs, y_train_fs, y_val_fs, y_test_fs




