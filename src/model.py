from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocessing import Preprocessing
from src.features import Features

class Model:
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.features = Features()

    def baseline_model(self, x_train, y_train, x_test):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        return y_pred
    
    def baseline_model_evaluation(self, y_test, y_pred):
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }