from functools import lru_cache

import pandas as pd

from schemas import DatasetRowChurn

DATASET_PATH = "data/churn_dataset.csv"


@lru_cache
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def to_dataset_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    records = df.to_dict(orient="records")
    return [DatasetRowChurn(**record) for record in records]
