
import numpy as np
from collections import Counter


class KNNClassifier:

    def __init__(self, n_neighbors=5, metric="euclidean"):
        """
        Khởi tạo mô hình KNN.

        Parameters:
        n_neighbors: Số lượng láng giềng gần nhất (K).
        metric: Phương pháp tính khoảng cách.
                'euclidean' hoặc 'manhattan'.
        """

        self.n_neighbors = n_neighbors
        self.metric = metric

        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.is_fitted = False

    # 1. Huấn luyện mô hình
    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều.")

        if y.ndim != 1:
            raise ValueError("y phải là mảng 1 chiều.")

        if len(X) != len(y):
            raise ValueError("X và y phải có cùng số mẫu.")

        if len(X) == 0:
            raise ValueError("Dữ liệu huấn luyện không được rỗng.")

        if not np.isfinite(X).all():
            raise ValueError("X chứa giá trị NaN hoặc vô cực.")

        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors phải là số nguyên dương.")

        if self.n_neighbors > len(X):
            raise ValueError("K không được lớn hơn số mẫu huấn luyện.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.is_fitted = True

        return self

    # 2. Tính khoảng cách Euclidean
    def _euclidean_distance(self, x1, x2):

        return np.sqrt(np.sum((x1 - x2) ** 2))

    # 3. Tính khoảng cách Manhattan
    def _manhattan_distance(self, x1, x2):

        return np.sum(np.abs(x1 - x2))

    # 4. Lựa chọn phương pháp tính khoảng cách
    def _calculate_distance(self, x1, x2):

        if self.metric == "euclidean":
            return self._euclidean_distance(x1, x2)

        elif self.metric == "manhattan":
            return self._manhattan_distance(x1, x2)

        else:
            raise ValueError("Metric không được hỗ trợ.")

    # 5. Tìm K láng giềng gần nhất
    def kneighbors(self, X):

        if not self.is_fitted:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit().")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Số đặc trưng của X không hợp lệ.")

        if not np.isfinite(X).all():
            raise ValueError("X chứa giá trị NaN hoặc vô cực.")

        distances_list = []
        indices_list = []

        for x in X:

            distances = np.array([
                self._calculate_distance(x, x_train)
                for x_train in self.X_train
            ])

            # Sắp xếp khoảng cách tăng dần
            indices = np.argsort(distances)[:self.n_neighbors]

            distances_list.append(distances[indices])
            indices_list.append(indices)

        return np.array(distances_list), np.array(indices_list)

    # 6. Dự đoán nhãn
    def predict(self, X):

        distances, indices = self.kneighbors(X)

        predictions = []

        for neighbor_indices in indices:

            neighbor_labels = self.y_train[neighbor_indices]

            # Majority Voting
            counts = Counter(neighbor_labels)

            # Xử lý trường hợp hòa phiếu:
            # Chọn nhãn xuất hiện đầu tiên trong classes_
            prediction = max(
                self.classes_,
                key=lambda label: counts[label]
            )

            predictions.append(prediction)

        return np.array(predictions)

    # 7. Tính xác suất dự đoán
    def predict_proba(self, X):

        _, indices = self.kneighbors(X)

        probabilities = []

        for neighbor_indices in indices:

            neighbor_labels = self.y_train[neighbor_indices]

            prob = [
                np.mean(neighbor_labels == c)
                for c in self.classes_
            ]

            probabilities.append(prob)

        return np.array(probabilities)

    # 8. Tính Accuracy
    def score(self, X, y):

        y = np.asarray(y)

        y_pred = self.predict(X)

        if len(y) != len(y_pred):
            raise ValueError("Số lượng nhãn thực tế không khớp.")

        return np.mean(y_pred == y)

    # 9. Đánh giá mô hình
    def evaluate(self, X, y):

        y = np.asarray(y)
        y_pred = self.predict(X)

        if len(y) != len(y_pred):
            raise ValueError("Số lượng nhãn thực tế không khớp.")

        accuracy = np.mean(y_pred == y)

        return {
            "accuracy": accuracy,
            "y_true": y,
            "y_pred": y_pred
        }

    # 10. Lấy tham số mô hình
    def get_params(self):

        return {
            "n_neighbors": self.n_neighbors,
            "metric": self.metric
        }
