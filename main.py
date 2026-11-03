from fastapi import FastAPI

from routers import dataset
from schemas import FeatureVectorChurn

app = FastAPI()

app.include_router(dataset.router)


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features
