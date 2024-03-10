import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from zenml import step

from configs.etl_config import mean_encoding_columns, binary_encoding_columns
from src.etl.categorical_encoding import MeanEncoder, BinaryEncoder

@step(enable_cache = True)
def categorical_encoder(
    df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "encoded_df"],
    Annotated[list, "dict_list"]
    ]:
    """
    Apply relevant encoding to categorical columns
    
    Args :
    -------
        df : pd.DataFrame = dataframe
    
    Returns : 
    ----------
        encoded_df : pd.DataFrame = Encoded dataframe
        dict_list : list = mapped data dictionary 
    """
    try :
        mean_encoder = MeanEncoder()
        encoded_df, mean_data_dictionary = mean_encoder.fit_transform(df, mean_encoding_columns)
        
        binary_encoder = BinaryEncoder()
        encoded_df, binary_data_dictionary = binary_encoder.fit_transform(encoded_df, binary_encoding_columns)
        
        dict_list = [mean_data_dictionary, binary_data_dictionary]
        print(encoded_df)
        logging.info("Successfully encoded the data")
        return encoded_df, dict_list
    except Exception as e :
        logging.error("Failed to categorical encode the data")
        raise e