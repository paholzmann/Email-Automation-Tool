import string
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import Logger

class Preprocessing:
    def __init__(self):
        self.logger = Logger(name="Preprocessing", level=logging.DEBUG).logger

    def lowercase_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Lowercasing texts")
        df["text_clean"] = df["text"].str.lower()
        return df
    
    def remove_text_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Removing punctuation from texts")
        df["text_clean"] = df["text_clean"].str.replace(r"[^\w\s$€]", "", regex=True)
        return df
    
    def whitespace_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug(f"Removing double whitespaces from texts")
        df["text_clean"] = df["text_clean"].str.replace(r"\s+", " ", regex=True)
        return df
    
    def split_train_test_data(self, df):
        self.logger.debug(f"Splitting train and test data")
        target = df["category"]
        features = df.drop(columns=["category"])
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test