from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "MLflow API Server"}

@app.post("/predict")
def predict(data: IrisInput):
    return {
        "class_name": "setosa"
    }