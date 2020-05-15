from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from pathlib import Path


def load_dataset(feature_file_path):
    feature_df = pd.read_csv(feature_file_path)
    feature_cols = filter(lambda x: "f" in x, feature_df.columns)
    labels = feature_df["label"].to_numpy()
    features = feature_df[feature_cols].to_numpy()
    return features, labels


def classify_slides(model_name, verbose=True):
    file_root = Path(f"./results/{model_name}")
    X_train, y_train = load_dataset(file_root / f"training_features.csv")
    X_test, y_test = load_dataset(file_root / f"validation_features.csv")

    classfier = RandomForestClassifier(random_state=42, n_estimators=5)
    classfier.fit(X_train, y_train)
    y_train_pred = classfier.predict(X_train)
    y_train_proba = classfier.predict_proba(X_train)[:, 1]
    y_test_pred = classfier.predict(X_test)
    y_test_proba = classfier.predict_proba(X_test)[:, 1]

    if verbose:
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Train acc: {train_acc}\tTest acc: {test_acc}.")

        train_auc = roc_auc_score(y_train, y_train_proba, average="weighted")
        test_auc = roc_auc_score(y_test, y_test_proba, average="weighted")
        print(f"Train AUC: {train_auc}\tTest AUC: {test_auc}.")

    validation_results = pd.DataFrame({"predictions": y_test_pred,
                                       "predict_proba": y_test_proba,
                                       "ground_truth": y_test})

    validation_results.to_csv(file_root / "validation_clf_results.csv")
