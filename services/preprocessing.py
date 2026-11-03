from functools import lru_cache

import pandas as pd
from sklearn.model_selection import train_test_split

from services.dataset import load_dataset

TARGET = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for column in NUMERIC_FEATURES:
        X[column] = X[column].fillna(X[column].median())
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna(X[column].mode()[0])
    return X


@lru_cache
def prepare_train_test(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_dataset()
    X, y = split_features_target(df)
    X = handle_missing_values(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def churn_distribution(y: pd.Series) -> dict:
    counts = y.value_counts()
    return {str(target_class): int(count) for target_class, count in counts.items()}
