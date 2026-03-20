import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from src.logger import Logger
from scipy.sparse import hstack

class Features:
    def __init__(self):
        self.logger = Logger(name="Features", level=logging.DEBUG).logger
    
    def term_frequency_inverse_document_frequency(self, x_train, x_test):
        n_docs = len(x_train)
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            min_df=1,
            max_df=1.0
            )
        x_train = tfidf.fit_transform(x_train["text_clean"])
        x_test = tfidf.transform(x_test["text_clean"])
        return x_train, x_test
    
    def count_text_length(self, x_train, x_test):
        x_train["text_length"] = x_train["text_clean"].str.len()
        x_test["text_length"] = x_test["text_clean"].str.len()
        return x_train, x_test
    
    def count_words(self, x_train, x_test):
        x_train["word_count"] = x_train["text_clean"].str.split().apply(len)
        x_test["word_count"] = x_test["text_clean"].str.split().apply(len)
        return x_train, x_test
    
    def count_avg_word_length(self, x_train, x_test):
        x_train["avg_word_length"] = x_train["text_clean"].apply( lambda x: sum(len(w) for w in x.split()) / len(x.split()) if x.split() else 0.0)
        x_test["avg_word_length"] = x_test["text_clean"].apply( lambda x: sum(len(w) for w in x.split()) / len(x.split()) if x.split() else 0.0)
        return x_train, x_test
    
    def get_num_digits(self, x_train, x_test):
        x_train["num_digits"] = x_train["text_clean"].str.count(r"\d")
        x_test["num_digits"] = x_test["text_clean"].str.count(r"\d")
        return x_train, x_test
    
    def get_num_exclemation(self, x_train, x_test):
        x_train["num_exclemation"] = x_train["text"].str.count("!")
        x_test["num_exclemation"] = x_test["text"].str.count("!")
        return x_train, x_test
    
    def get_num_questions(self, x_train, x_test):
        x_train["num_questions"] = x_train["text"].str.count(r"\?")
        x_test["num_questions"] = x_test["text"].str.count(r"\?")
        return x_train, x_test
    
    def create_statistical_x(self, x_train, x_test):
        x_train = x_train.drop(columns=["text" ,"text_clean"])
        x_test = x_test.drop(columns=["text", "text_clean"])
        return x_train, x_test
    
    def combine_tfidf_and_statisticals(self, x_tfidf, x_statistical):
        return hstack([x_tfidf, x_statistical.values])