from src.file_handler import FileHandler
from src.preprocessing import Preprocessing
from src.model import Model
from src.pipeline import WorkflowPipeline
from src.features import Features
from src.report import Report
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()
file_handler = FileHandler()
preprocessor = Preprocessing()
model = Model()
workflow_pipeline = WorkflowPipeline()
features = Features()
report = Report()

path_configs = file_handler.load_configs(filepath="configs/paths.json")
baseline_model_exists = model.check_if_model_exists(model_path="models/baseline_model.pkl")
if not baseline_model_exists:
    df = file_handler.load_data(filepath=path_configs["train_test_data"])
    clean_df = workflow_pipeline.preprocessing(df=df)

    x_train, x_test, y_train, y_test = preprocessor.split_train_test_data(df=clean_df)
    y_pred, pipeline = model.baseline_model(x_train=x_train["text_clean"], y_train=y_train, x_test=x_test["text_clean"])
    baseline_evaluation = model.baseline_model_evaluation(y_test=y_test, y_pred=y_pred)
    report.visualize_confusion_matrix(model_name="baseline_model", confusion_matrix=baseline_evaluation["confusion_matrix"])
    report.visualize_classification_report(model_name="baseline_model", classification_report=baseline_evaluation["classification_report"])
    model.save_model(pipeline=pipeline, model_path="models/baseline_model.pkl")

baseline_model = model.load_model(model_path="models/baseline_model.pkl")


class Request(BaseModel):
    text: str

@app.post("/predict")
def make_model_prediction(request: Request):
    """
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"I WANT A REFUND NOW!\"}
    """
    prediction = baseline_model.predict([request.text])[0]
    return {"prediction": prediction}