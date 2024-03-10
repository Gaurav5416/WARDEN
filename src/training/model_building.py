from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier



class Model(ABC):
    """Abstract class for models."""

    @abstractmethod
    def train(self, X_train : pd.DataFrame, y_train : pd.Series):
        """Trains the model."""
        pass

class Adaboost(Model) :
    """Model class that uses sklearn adaptive boosting to train the model"""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Args:
        X_train: pandas DataFrame, the training features.
        y_train: pandas Series, the training target.
        """
        self.X_train = X_train
        self.y_train = y_train

    def train(self)-> AdaBoostClassifier:
        """
        Trains the model using sklearn's Adaptive boosting

        Parameters
        -----------
            X_train : pd.DataFrame 
            y_train : pd.Series

        Results 
        --------
            model : Trained model
        """
        adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), 
                                      random_state=42, 
                                      algorithm='SAMME')
        
        model = adaboost.fit(self.X_train, self.y_train)
        
        return model

class TrainTestSplitter:
    """A class used to split data into training and testing parts"""

    def __init__(self, df: pd.DataFrame, features: List[str], target: str, test_size: float = 0.2):
        """
        Args:
        df: pandas DataFrame, the entire dataset.
        features: list of str, the column names to be used as features.
        target: str, the column name to be used as target.
        test_size: float, proportion of the dataset to include in the test split.
        """
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data into training and testing parts.
        
        Returns:
        Tuple of pandas DataFrame and Series: (X_train, X_test, y_train, y_test)
        """ 
        # drop month_year and id columns 
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        return X_train, X_test, y_train, y_test




class DataSplitter: 
    """A class used to split data into 50-50 sample into reference and current dataset"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
        df: pandas DataFrame, the entire dataset.
        """
        self.df = df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into reference and current dataset
        
        Returns:
        Tuple of pandas DataFrames: (reference, current)
        """ 
        shuffled_df = self.df.sample(frac=1)
        reference = shuffled_df.iloc[:len(shuffled_df)//2]
        current = shuffled_df.iloc[len(shuffled_df)//2:]
        return reference, current

