from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

class Encoder(ABC):
    """
    The Encoder interface declares operations common to all supported encoding algorithms.
    """
    
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, cols:list[str]) -> Union[pd.DataFrame, dict]:
        """
        Fits and transform the encoder to provided dataframe.

        Parameters:
        ------------
                df : pd.DataFrame = DataFrame to fit the encoder on.
                col : str = Column to fit the encoder on.

        Returns:
        ---------
        df : pd.DataFrame =  Transformed DataFrame.   
        data_dictionary : dict = data dictionary for reverse mapping 
        """
        pass

class MeanEncoder(Encoder):
    """
    Encoding strategy which uses mean encoding on the data. 

    """
    def fit_transform(self, df: pd.DataFrame, cols:list[str]) -> Union[pd.DataFrame, dict]:
        """
        Fits and transform the provided dataframe using mean encoding

        Parameters:
        ------------
                df : pd.DataFrame = DataFrame to fit the encoder on.
                col : str = Column to fit the encoder on.

        Returns:
        ---------
        df : pd.DataFrame =  Transformed DataFrame.   
        data_dictionary : dict = data dictionary for reverse mapping 

        """
        data_dictionaries = {}
        for col in cols:
            fraud_mean = df.groupby(col)['is_fraud'].mean().round(2)
            df[col] = df[col].map(fraud_mean)
            data_dictionaries[col] = fraud_mean.to_dict()
        return df, data_dictionaries
    
class BinaryEncoder(Encoder):
    """
    Encoding strategy which uses binary encoding on the data. 

    """
    def fit_transform(self, df: pd.DataFrame, cols:list[str]) -> Union[pd.DataFrame, dict]:
        """
        Fits and transform the provided dataframe using binary encoding

        Parameters:
        ------------
                df : pd.DataFrame = DataFrame to fit the encoder on.
                col : str = Column to fit the encoder on.

        Returns:
        ---------
        df : pd.DataFrame =  Transformed DataFrame.   
        data_dictionary : dict = data dictionary for reverse mapping 
        
        """
        data_dictionaries = {}
        for col in cols:
    
            encoded_values = pd.get_dummies(df[col], prefix='gender', drop_first=True)
            mapping_dict = {value: f'{col} = {value}' for value in df[col].unique()}
            data_dictionaries[col] = mapping_dict
            df = pd.concat([df.drop(col, axis=1), encoded_values], axis=1)
            
        return df, data_dictionaries
