import os
import logging
import pandas as pd
from dotenv import load_dotenv
from typing_extensions import Annotated

from zenml import step

from src.etl.data_ingestion import DataLoader
from configs.etl_config import table_name

@step(enable_cache = True)
def data_loader(
    table_name:str = table_name,
    local_storage : bool = True,
    for_predict : bool = False,
    )-> Annotated[pd.DataFrame, "df"]:
    """Read data from SQL table and return a pandas dataframe
    
    Args :
        table_name : Name of the table to read from
    
    Returns : 
        Pandas.DataFrame"""
     
    try:
        if local_storage == True :
            df = pd.read_csv("data/fraud.csv")
            df.drop(columns= ["Unnamed: 0"], axis = 1, inplace=True)
        else : 
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

