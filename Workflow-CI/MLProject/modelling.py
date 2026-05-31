from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


TARGET = "default_payment_next_month"


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return (
        train.drop(columns=[TARGET]),
        train[TARGET],
        test.drop(columns=[TARGET]),
        test[TARGET],
    )


def parse_depth(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="credit_default_preprocessing")
    parser.add_argument("--max-depth", default="12")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_dataset(Path(args.data_dir))
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=parse_depth(args.max_depth),
        random_state=42,
        n_jobs=-1,
    )

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        (artifact_dir / "metric_info.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No default", "Default"], cmap="Blues")
        plt.tight_layout()
        plt.savefig(artifact_dir / "training_confusion_matrix.png", dpi=160)
        plt.close()
        mlflow.log_artifacts(str(artifact_dir))
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(5).astype(float))

        Path("latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")


if __name__ == "__main__":
    main()
