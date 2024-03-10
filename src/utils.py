from zenml.client import Client
from zenml import step
client = Client()
import pandas as pd
from typing_extensions import Annotated
def mapped_spitter(category:str) :
    training_pipeline = client.get_pipeline("training_warden")
    last_run = training_pipeline.last_run
    dict_list = last_run.steps["categorical_encoder"].outputs["dict_list"].load()

    category_double_dict = dict_list[0]
    category_dict = category_double_dict["category"]
    mapped_category_value = category_dict[category]
    return mapped_category_value

@step(enable_cache = False)
def get_reference_data()-> Annotated[pd.DataFrame, "reference_data"]:
    """
    Bring last deployed model data for testing purposes

    Args :
    ------
    None

    Results :
    ---------
    pandas.DataFrame : Reference data
    """
    
    training_pipeline = client.get_pipeline("inference_pipeline")
    last_run = training_pipeline.last_run
    reference_data = last_run.steps["bentoml_predictor"].outputs["batch_data"].load()
    return reference_data

