import logging
import pandas as pd 
from datetime import datetime
from typing_extensions import Annotated
from typing import Tuple
from zenml import step
from zenml.client import Client
from src.utils import mapped_spitter

@step(enable_cache=False)
def dynamic_importer()-> Annotated[pd.DataFrame, "batch_1"] :
    try:

        client = Client()
        training_pipeline = client.get_pipeline("training_warden")
        last_run = training_pipeline.last_run
        
        # batch_1 = last_run.steps["model_evaluator"].outputs["batch_data"].load()
        batch_2 = last_run.steps["data_splitter"].outputs["batch_2"].load()
        # Sampling 20% of data to match batch_1 standards 
        batch_2 = batch_2.sample(frac=0.2, random_state=42)
        return batch_2
    
    except Exception as e:
        
        logging.exception(e)
        return e
