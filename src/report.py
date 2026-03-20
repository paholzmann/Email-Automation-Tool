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

    def visualize_confusion_matrix(self, confusion_matrix):
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True, cmap="Blues")
        plt.show()