from abc import ABC, abstractmethod
import pandas as pd

class DataBalancer(ABC):

    """
    Abstract Class defining strategy for balancing data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        Abstract method that balanced the class distribution

        Parameters:
        ------------
                df : pd.DataFrame = DataFrame to balance

        Returns:
        ---------
        df : pd.DataFrame =  Transformed DataFrame.   

        """
        pass


class Undersampler(DataBalancer):

    """
    Data balancing strategy which undersamples the data.
    """

    def handle_data(self, df: pd.DataFrame, target_col : str) -> pd.DataFrame:
        """
        Removes columns which are not required, undersamples the data to balance class ratio
        
        Args :
        ------
            pd.DataFrame : dataframe
            target_col : target column of the dataframe

        Returns :
        ---------
            pd.DataFrame: balanced dataframe
        
        """
        try:
            df = df.sample(frac=1)
            num_fraud_instances = df[target_col].value_counts()[1]
            fraud_df = df.loc[df[target_col] == 1]
            non_fraud_df = df.loc[df[target_col] == 0][:num_fraud_instances]
            normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

            balanced_df = normal_distributed_df.sample(frac=1, random_state=42)
            return balanced_df
        except Exception as e:
            raise e