import os
import gzip
import struct
import numpy as np

from sklearn.model_selection import train_test_split


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "EMNIST", "raw")
)


# ==========================================
# 2. ĐỌC DỮ LIỆU IDX
# ==========================================

def open_idx_file(filepath):
    """
    Mở file IDX ở dạng thông thường hoặc gzip (.gz).
    """
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rb")

    return open(filepath, "rb")


def load_images(filepath):
    """
    Đọc file ảnh IDX3.
    Trả về mảng numpy có dạng (N, 28, 28).
    """
    with open_idx_file(filepath) as f:
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
    with open_idx_file(filepath) as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("File nhan IDX khong hop le!")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

        if len(labels) != num_labels:
            raise ValueError("So luong nhan khong khop voi header IDX!")

    return labels


# ==========================================
# 3. LOAD TOÀN BỘ DATASET
# ==========================================

def get_emnist_filepath(split, data_type):
    """
    Tạo đường dẫn file EMNIST.

    split:
        balanced, byclass, bymerge, letters, digits hoặc mnist.

    data_type:
        train-images, train-labels, test-images hoặc test-labels.
    """
    filename = f"emnist-{split}-{data_type}-idx"

    if "images" in data_type:
        filename += "3-ubyte"
    else:
        filename += "1-ubyte"

    filepath = os.path.join(RAW_DIR, filename)

    # Hỗ trợ cả file đã giải nén và file .gz.
    if os.path.exists(filepath):
        return filepath

    if os.path.exists(filepath + ".gz"):
        return filepath + ".gz"

    raise FileNotFoundError(
        f"Khong tim thay file: {filepath} hoac {filepath}.gz"
    )


def load_emnist(split="balanced", transpose=True):
    """
    Đọc bộ dữ liệu EMNIST.

    split:
        Chọn phiên bản EMNIST:
        balanced, byclass, bymerge, letters, digits hoặc mnist.

    transpose=True:
        Chuyển vị hai chiều ảnh (28, 28) để hiệu chỉnh
        cách lưu ảnh thường gặp trong bộ EMNIST.

    Lưu ý:
        Nên trực quan hóa ảnh sau khi load để xác nhận hướng ảnh.
    """

    X_train = load_images(
        get_emnist_filepath(split, "train-images")
    )

    y_train = load_labels(
        get_emnist_filepath(split, "train-labels")
    )

    X_test = load_images(
        get_emnist_filepath(split, "test-images")
    )

    y_test = load_labels(
        get_emnist_filepath(split, "test-labels")
    )

    # Kiểm tra số lượng ảnh và nhãn.
    if len(X_train) != len(y_train):
        raise ValueError("So luong anh train va nhan train khong khop!")

    if len(X_test) != len(y_test):
        raise ValueError("So luong anh test va nhan test khong khop!")

    # Hiệu chỉnh hướng ảnh EMNIST nếu cần.
    if transpose:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, y_train, X_test, y_test


# ==========================================
# 4. TIỀN XỬ LÝ DỮ LIỆU
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
# 5. PIPELINE TIỀN XỬ LÝ
# ==========================================

def preprocess_emnist(flatten=True,
                      val_size=0.2,
                      split="balanced",
                      transpose=True):
    """
    Pipeline tiền xử lý EMNIST.

    flatten=True:
        Ảnh được chuyển thành vector 784 chiều.

    flatten=False:
        Giữ nguyên ảnh 28x28.

    split:
        Phiên bản EMNIST, mặc định là balanced.

    transpose=True:
        Chuyển vị ảnh để hiệu chỉnh hướng ảnh.
    """

    # Load raw data
    X_train_em, y_train_em, X_test_em, y_test_em = load_emnist(
        split=split,
        transpose=transpose
    )

    # Normalize
    X_train_em = normalize_data(X_train_em)
    X_test_em = normalize_data(X_test_em)

    # Flatten nếu cần
    if flatten:
        X_train_em = flatten_data(X_train_em)
        X_test_em = flatten_data(X_test_em)

    # Split train / validation
    X_train_em, X_val_em, y_train_em, y_val_em = split_validation(
        X_train_em,
        y_train_em,
        val_size=val_size
    )

    print(f"===== DATASET EMNIST ({split.upper()}) INFORMATION =====")
    print("Train:", X_train_em.shape, y_train_em.shape)
    print("Validation:", X_val_em.shape, y_val_em.shape)
    print("Test:", X_test_em.shape, y_test_em.shape)
    print("Number of classes:", len(np.unique(y_train_em)))
    print("Pixel range:", X_train_em.min(), "-", X_train_em.max())

    return (
        X_train_em,
        X_val_em,
        X_test_em,
        y_train_em,
        y_val_em,
        y_test_em
    )
