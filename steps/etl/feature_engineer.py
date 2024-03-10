import logging
import pandas as pd
from typing_extensions import Annotated

from zenml import step

from src.etl.feature_engineering import DateFeatureEngineer, AgeFeatureEngineer
from configs.etl_config import feature_engineering_columns, cols_to_drop


@step(enable_cache = True)
def feature_engineer(
    df: pd.DataFrame)-> Annotated[pd.DataFrame, "transformed_df"] :
    """
    Performs feature engineering on the data.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be processed.
        
    Returns:
        pd.DataFrame: DataFrame after feature engineering.
    """
    try:

        df.drop(columns=cols_to_drop, axis=1, inplace=True)
        
        date_engineer = DateFeatureEngineer()
        transformed_df = date_engineer.fit_transform(df, feature_engineering_columns[0])
        
        age_engineer = AgeFeatureEngineer()
        transformed_df = age_engineer.fit_transform(transformed_df, feature_engineering_columns[1])

        logging.info("Feature engineering applied successfully") 
        return transformed_df

    except Exception as e:
        
        logging.error(e)
        raise e
