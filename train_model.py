import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Employee_Attrition_Data.csv"
MODEL_PATH = BASE_DIR / "attrition_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates()
    return data


def build_pipeline(numeric_columns, categorical_columns):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])

    classifier = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def train_model():
    print("Loading dataset...")
    data = load_dataset()
    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    # 1 = employee leaves (Attrition = Yes), 0 = employee stays
    y = data["Attrition"].map({"No": 0, "Yes": 1})
    X = data.drop(columns=["Attrition", "EmployeeID"])

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    print("Input features:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = build_pipeline(numeric_columns, categorical_columns)

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("Recall  :", round(recall, 4))
    print("ROC-AUC :", round(roc_auc, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved as attrition_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
