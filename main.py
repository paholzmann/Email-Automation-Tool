from src.file_handler import FileHandler
from src.preprocessing import Preprocessing
from src.model import Model
from src.pipeline import WorkflowPipeline
from src.features import Features
from src.report import Report

if __name__ == "__main__":
    file_handler = FileHandler()
    preprocessor = Preprocessing()
    model = Model()
    workflow_pipeline = WorkflowPipeline()
    features = Features()
    report = Report()
    path_configs = file_handler.load_configs(filepath="configs/paths.json")
    df = file_handler.load_data(filepath=path_configs["train_test_data"])
    clean_df = workflow_pipeline.preprocessing(df=df)
    x_train, x_test, y_train, y_test = preprocessor.split_train_test_data(df=clean_df)
    y_pred, pipeline = model.baseline_model(x_train=x_train["text_clean"], y_train=y_train, x_test=x_test["text_clean"])
    baseline_evaluation = model.baseline_model_evaluation(y_test=y_test, y_pred=y_pred)
    print(f"Accuracy: {baseline_evaluation['accuracy']}")
    print(f"Classification report: \n {baseline_evaluation['classification_report']}")
    print(f"Confusion matrix: \n {baseline_evaluation['confusion_matrix']}")
    report.visualize_confusion_matrix(model_name="baseline_model", confusion_matrix=baseline_evaluation["confusion_matrix"])
    report.visualize_classification_report(model_name="baseline_model", classification_report=baseline_evaluation["classification_report"])
    model.save_baseline_model(pipeline=pipeline)