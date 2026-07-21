import os
import pandas as pd


DATASETS_INFO = {
    "CM1": {"file": "CM1.csv"},
    "JM1": {"file": "JM1.csv"},
    "KC1": {"file": "KC1.csv"},
    "KC3": {"file": "KC3.csv"},
    "KC4": {"file": "KC4.csv"},
    "MC1": {"file": "MC1.csv"},
    "MC2": {"file": "MC2.csv"},
    "PC1": {"file": "PC1.csv"},
    "PC2": {"file": "PC2.csv"},
    "PC3": {"file": "PC3.csv"},
    "PC4": {"file": "PC4.csv"},
    "PC5": {"file": "PC5.csv"},
}


POSSIBLE_TARGET_COLUMNS = [
    "Defective",
    "defective",
    "label",
    "Label",
    "class",
    "Class",
    "bug",
    "Bug",
    "defects",
    "Defects",
]


DROP_COLUMNS = [
    "id",
    "ID",
    "name",
    "Name",
    "module",
    "Module",
    "file",
    "File",
]


def find_target_column(df):
    for col in POSSIBLE_TARGET_COLUMNS:
        if col in df.columns:
            return col

    return df.columns[-1]


def encode_target(y):
    y = y.copy()

    if y.dtype == object:
        y = y.astype(str).str.strip()

    values = set(y.dropna().unique())

    if values == {"N", "Y"}:
        return y.map({"N": 0, "Y": 1}).astype(int)

    if values == {"No", "Yes"}:
        return y.map({"No": 0, "Yes": 1}).astype(int)

    if values == {"false", "true"}:
        return y.map({"false": 0, "true": 1}).astype(int)

    if values == {"False", "True"}:
        return y.map({"False": 0, "True": 1}).astype(int)

    if values == {"FALSE", "TRUE"}:
        return y.map({"FALSE": 0, "TRUE": 1}).astype(int)

    if values == {"0", "1"}:
        return y.astype(int)

    if values == {0, 1}:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        return (y >= 1).astype(int)

    raise ValueError(f"Unknown target labels: {values}")


def clean_features(df, target_column):
    X = df.drop(columns=[target_column])

    cols_to_drop = [col for col in DROP_COLUMNS if col in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.mean(numeric_only=True))
    X = X.dropna(axis=1)

    return X


def load_dataset(dataset_name, datasets_dir="datasets"):
    dataset_name = dataset_name.upper()

    if dataset_name not in DATASETS_INFO:
        raise ValueError(f"Dataset {dataset_name} is not defined.")

    file_path = os.path.join(
        datasets_dir,
        DATASETS_INFO[dataset_name]["file"]
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    target_column = find_target_column(df)
    X = clean_features(df, target_column)
    y = encode_target(df[target_column])

    return X, y, df, target_column


def load_all_datasets(datasets_dir="datasets"):
    all_data = {}

    for dataset_name in DATASETS_INFO.keys():
        X, y, df, target_column = load_dataset(dataset_name, datasets_dir)

        all_data[dataset_name] = {
            "X": X,
            "y": y,
            "df": df,
            "target_column": target_column,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "class_distribution": y.value_counts().to_dict(),
        }

    return all_data


def print_dataset_summary(dataset_name, datasets_dir="datasets"):
    X, y, df, target_column = load_dataset(dataset_name, datasets_dir)

    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Original shape: {df.shape}")
    print(f"X shape after cleaning: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target column: {target_column}")
    print("Class distribution:")
    print(y.value_counts())
    print("=" * 60)


def print_all_datasets_summary(datasets_dir="Datasets"):
    for dataset_name in DATASETS_INFO.keys():
        print_dataset_summary(dataset_name, datasets_dir)