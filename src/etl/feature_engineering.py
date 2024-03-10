from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class FeatureEngineer(ABC):
    """
    Abstract Base Class representing the Feature Engineer.
    """

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Fit and transform the DataFrame.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str) : Name of the column to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass


class DateFeatureEngineer(FeatureEngineer):
    """
    This class handles feature engineering for date type variables.
    """

    def fit_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Fit and transform the data.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): column name to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df = self._split_date(df, column)
        return df

    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Splits a date into separate features.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column in the dataframe to be transformed.
        
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        # Convert the column to datetime format
        df[column] = pd.to_datetime(df[column])

        # Extract date, month, and hour into separate columns
        df['trans_year'] = df[column].dt.year
        df['trans_month'] = df[column].dt.month
        df['trans_day'] = df[column].dt.day
        df['trans_hour'] = df[column].dt.hour
        df['trans_dayofweek'] = df[column].dt.dayofweek

        df.drop(column, axis=1, inplace=True)

        return df


class AgeFeatureEngineer(FeatureEngineer):
    """
    This class handles feature engineering for calculating age from date of birth.
    """

    def fit_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Fit and transform the data to calculate age from date of birth.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): Column name containing date of birth.

        Returns:
            pd.DataFrame: The transformed DataFrame with age calculated.
        """
        df[column] = pd.to_datetime(df[column])
        df['age'] = (datetime.now() - df[column]).astype('<m8[Y]')
        df.drop(columns=column, inplace=True)

        return df