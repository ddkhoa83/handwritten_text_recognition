# src/models/ensemble_model.py
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class MNISTStackingEnsemble:

  def __init__(self, cv_folds=5):
    # Định nghĩa 3 base models với các siêu tham số cơ bản
    base_estimators = [
        ('nb', GaussianNB()),
        (
            'mlp',
            MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42
            ),
        ),
        ('svc', SVC(probability=True, kernel='rbf', random_state=42)),
    ]

    # Khởi tạo mô hình Stacking với Meta-learner là Logistic Regression đa lớp
    self.stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv_folds,
        n_jobs=-1,  # Tận dụng tối đa đa luồng CPU khi train
    )

  def fit(self, X_train, y_train):
    """Huấn luyện mô hình Stacking trên tập train"""
    self.stacking_model.fit(X_train, y_train)
    return self

  def predict(self, X_test):
    """Dự đoán nhãn cho tập test"""
    return self.stacking_model.predict(X_test)

  def evaluate(self, X_test, y_test):
    """Độ chính xác trên tập test"""
    return self.stacking_model.score(X_test, y_test)