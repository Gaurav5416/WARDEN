import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from zenml import step

from configs.etl_config import target_col
from src.training.model_building import TrainTestSplitter, DataSplitter

@step(enable_cache = False)
def train_test_splitter( 
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    ]:
    
    """Splits data into training and testing parts.""" 
    
    try:
        data_splitter = TrainTestSplitter(df, features = df.drop(target_col, axis=1).columns, target=target_col) 
        X_train, X_test, y_train, y_test = data_splitter.split() 
        logging.info("Data split successfully")
        return X_train, X_test, y_train, y_test  
    
    except Exception as e:
        logging.error(e)
        raise e

@step(enable_cache = False)
def data_splitter( 
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "batch_1"],
    Annotated[pd.DataFrame, "batch_2"],
    ]:
    
    """Splits data into reference and current dataset""" 
    
    try:

        data_splitter = DataSplitter(df) 
        batch_1, batch_2 = data_splitter.split() 
        logging.info("Data splitted successfully")
        return batch_1, batch_2
    
    except Exception as e:
        
        logging.error(e)
        raise e
