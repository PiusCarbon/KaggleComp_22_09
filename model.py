from abc import ABC, abstractmethod
from random import Random
import pandas as pd


class Model(ABC):

    @abstractmethod
    def train(self, df_attributes: pd.DataFrame, s_labels: pd.Series):
        pass
    @abstractmethod
    def predict(self, df_attributes: pd.DataFrame) -> pd.Series:
        pass
    def evaluate(self, s_true: pd.Series, s_predicted: pd.Series) -> float:
        return 100 / s_true.size * (2 * (s_predicted - s_true).abs() / s_true.abs().add(s_predicted.abs())).sum()


from sklearn.ensemble import RandomForest 
class RandomForestModel(Model):

    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForest(seed=42)
        

    def train(self, df_attributes: pd.DataFrame, s_labels: pd.Series):
        self.fit(df_attributes, s_labels)

    def predict(self, df_attributes: pd.DataFrame) -> pd.Series:
        return self.model.predict(df_attributes)
