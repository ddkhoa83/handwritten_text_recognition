import sys
from pathlib import Path

#Get the project root:
#main.py -> src -> handwritten_text_recognition.
project_dir: Path = Path(__file__).resolve().parent.parent

#Allow Python to find modules inside the project root.
sys.path.insert(0, str(project_dir))

from data.preprocessing import (
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test,
)
from Logistic_Regression import LogisticRegression

if __name__ == "__main__":
    "==============Logistic Regression==============="
    #Create the model.
    model: LogisticRegression = LogisticRegression(
        learning_rate = 0.1,
        n_epochs = 30,
        batch_size = 256,
        reg_strength = 0.0001,
        random_state = 42,
    )
    #Training using the training subset.
    model.fit(X_train_scaled, y_train)

    #Evaluate using the validation subset.
    validation_accuracy: float = model.score(X_val_scaled, y_val)

    #Display validation accuracy.
    print(f"Validation accuracy: {validation_accuracy: .2%}")