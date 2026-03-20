import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from src.logger import Logger
import pandas as pd
from pathlib import Path
import os

class Report:
    def __init__(self):
        self.logger = Logger(name="Report", level=logging.DEBUG).logger

    def visualize_classification_report(self, model_name, classification_report):
        df_report = pd.DataFrame(classification_report).transpose()
        plt.figure()
        sns.heatmap(df_report.iloc[:, :-1], annot=True)
        plt.title(f"{model_name} Classification Report Heatmap")
        plt.savefig(f"reports/classification_report_{model_name}")

    def visualize_confusion_matrix(self, model_name, confusion_matrix):
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(f"reports/confusion_matrix_{model_name}")