import logging
import pandas as pd
from typing_extensions import Annotated

from zenml import step

from src.etl.balance_data import Undersampler
from configs.etl_config import target_col

@step(enable_cache = True)
def data_balancer(
    df: pd.DataFrame,
    )-> Annotated[pd.DataFrame, "balanced_df"] :
    try :
        undersampler = Undersampler()
        balanced_df = undersampler.handle_data(df, target_col)
        logging.info("Successfully undersampled the data")
        return balanced_df
    except Exception as e :
        logging.error("Failed to undersample the data")
        raise e