from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

class MNISTStackingEnsemble:
    def __init__(self, *models, final_estimator=None, cv_folds=5):
        """
        Nhận vào số lượng mô hình cơ sở tùy ý thông qua *models.
        Mỗi phần tử có thể là một instance mô hình (ví dụ: GaussianNB()) 
        hoặc một tuple dạng ('tên_goi', model_instance).
        """
        estimators = []
        for i, m in enumerate(models):
            if isinstance(m, tuple) and len(m) == 2:
                estimators.append(m)
            else:
                # Tự động tạo tên định danh dựa trên tên class của mô hình
                name = f"{type(m).__name__.lower()}_{i}"
                estimators.append((name, m))
        
        # Nếu không truyền meta-learner riêng, mặc định dùng Logistic Regression đa lớp
        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000)

        # Khởi tạo StackingClassifier
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv_folds,
            n_jobs=-1
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