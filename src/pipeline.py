import pandas as pd
import logging
import joblib
from src.preprocessing import Preprocessing
from src.model import Model
from src.logger import Logger
from src.features import Features
from sklearn.metrics import accuracy_score, classification_report

class WorkflowPipeline:
    def __init__(self):
        self.logger = Logger(name="Pipeline", level=logging.DEBUG).logger
        self.preprocessor = Preprocessing()
        self.model = Model()
        self.features = Features()

    def preprocessing(self, df):
        df = self.preprocessor.lowercase_texts(df=df)
        df = self.preprocessor.remove_text_punctuation(df=df)
        df = self.preprocessor.whitespace_normalization(df=df)
        return df

    def feature_engineering(self):
        pass

    def model_training(self):
        pass