import os
import logging
import pandas as pd
from dotenv import load_dotenv
from typing_extensions import Annotated

from zenml import step

from src.etl.data_ingestion import DataLoader

@step(enable_cache = True)
def data_loader(
    table_name:str,
    for_predict : bool = False,
    )-> Annotated[pd.DataFrame, "df"]:
    """Read data from SQL table and return a pandas dataframe
    
    Args :
        table_name : Name of the table to read from
    
    Returns : 
        Pandas.DataFrame"""
     
    try:
        load_dotenv()
        db_url = os.getenv('DB_URL')
        data_loader = DataLoader(db_url)
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        if for_predict :
            df.drop(columns = ['is_fraud'], inplace = True)
        logging.info(f"Successfully ingest data from {table_name}")
        return df
    
    except Exception as e :
        logging.error(f"Error reading data from {table_name}")
        raise e

