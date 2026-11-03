from fastapi import APIRouter, Query

from services.dataset import load_dataset, to_dataset_rows
from services.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    churn_distribution,
    prepare_train_test,
)
from schemas import DatasetRowChurn

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def preview(n: int = Query(default=5, ge=1, le=200)) -> list[DatasetRowChurn]:
    df = load_dataset()
    return to_dataset_rows(df.head(n))


@router.get("/info")
def info() -> dict:
    df = load_dataset()
    churn_counts = df["churn"].value_counts()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "features": [column for column in df.columns if column != "churn"],
        "churn_distribution": {
            str(target_class): int(count)
            for target_class, count in churn_counts.items()
        },
    }


@router.get("/split-info")
def split_info() -> dict:
    X_train, X_test, y_train, y_test = prepare_train_test()
    return {
        "train": {
            "rows": len(X_train),
            "churn_distribution": churn_distribution(y_train),
        },
        "test": {
            "rows": len(X_test),
            "churn_distribution": churn_distribution(y_test),
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
