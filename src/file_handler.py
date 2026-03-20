import pandas as pd
import json

class FileHandler:
    def __init__(self):
        pass

    def load_configs(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)