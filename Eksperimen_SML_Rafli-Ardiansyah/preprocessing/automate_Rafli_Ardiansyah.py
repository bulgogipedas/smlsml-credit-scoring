from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
TARGET = "default_payment_next_month"
CATEGORICAL_COLUMNS = ["sex", "education", "marriage"]
ID_COLUMNS = ["id"]


def _clean_columns(columns: list[str]) -> list[str]:
    cleaned = []
    for column in columns:
        value = str(column).strip().lower()
        value = value.replace(" ", "_").replace("-", "_")
        value = value.replace(".", "").replace("/", "_")
        if value == "default_payment_next_month":
            value = TARGET
        cleaned.append(value)
    return cleaned


def load_raw_data(source: str | None = None) -> pd.DataFrame:
    path_or_url = source or DATA_URL
    df = pd.read_excel(path_or_url, header=1)
    df.columns = _clean_columns(list(df.columns))
    return df


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_columns(list(df.columns))

    if "default_payment_next_month" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": TARGET})
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")

    df = df.drop(columns=[col for col in ID_COLUMNS if col in df.columns])
    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])]

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int).astype(str)

    numeric_cols = [col for col in df.columns if col not in CATEGORICAL_COLUMNS + [TARGET]]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def preprocess(df: pd.DataFrame, output_dir: Path, test_size: float = 0.2) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_credit_data(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    train = pd.DataFrame(X_train_arr, columns=feature_names)
    train[TARGET] = y_train.to_numpy()
    test = pd.DataFrame(X_test_arr, columns=feature_names)
    test[TARGET] = y_test.to_numpy()

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    metadata = {
        "dataset": "UCI Default of Credit Card Clients",
        "source": DATA_URL,
        "target": TARGET,
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_names)),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "positive_rate": float(y.mean()),
    }
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def make_sample_data(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "limit_bal": rng.integers(10_000, 500_000, rows),
            "sex": rng.choice([1, 2], rows),
            "education": rng.choice([1, 2, 3, 4], rows),
            "marriage": rng.choice([1, 2, 3], rows),
            "age": rng.integers(21, 75, rows),
            "pay_0": rng.integers(-2, 8, rows),
            "pay_2": rng.integers(-2, 8, rows),
            "pay_3": rng.integers(-2, 8, rows),
            "pay_4": rng.integers(-2, 8, rows),
            "pay_5": rng.integers(-2, 8, rows),
            "pay_6": rng.integers(-2, 8, rows),
            "bill_amt1": rng.normal(50_000, 30_000, rows),
            "bill_amt2": rng.normal(48_000, 30_000, rows),
            "bill_amt3": rng.normal(45_000, 30_000, rows),
            "bill_amt4": rng.normal(42_000, 30_000, rows),
            "bill_amt5": rng.normal(40_000, 30_000, rows),
            "bill_amt6": rng.normal(38_000, 30_000, rows),
            "pay_amt1": rng.normal(5_000, 4_000, rows),
            "pay_amt2": rng.normal(5_000, 4_000, rows),
            "pay_amt3": rng.normal(4_500, 4_000, rows),
            "pay_amt4": rng.normal(4_000, 4_000, rows),
            "pay_amt5": rng.normal(3_500, 4_000, rows),
            "pay_amt6": rng.normal(3_500, 4_000, rows),
        }
    )
    risk = 0.2 + (df["pay_0"] > 1) * 0.25 + (df["limit_bal"] < 80_000) * 0.15
    df[TARGET] = rng.binomial(1, np.clip(risk, 0.05, 0.85))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="Sumber XLS/CSV lokal opsional. Default menggunakan URL UCI.")
    parser.add_argument("--output-dir", default="SMSML_Rafli-Ardiansyah/Membangun_model/credit_default_preprocessing")
    parser.add_argument("--sample", action="store_true", help="Gunakan data contoh deterministik untuk smoke test offline.")
    args = parser.parse_args()

    if args.sample:
        df = make_sample_data()
    elif args.source and args.source.endswith(".csv"):
        df = pd.read_csv(args.source)
    else:
        df = load_raw_data(args.source)

    metadata = preprocess(df, Path(args.output_dir))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
