import json
import logging
import pandas as pd
from typing_extensions import Annotated

from steps.monitoring.alerter import warden_slackbot

from zenml import step
from zenml.client import Client
client = Client()




def mapped_spitter(category:str) :
    """
    Takes a category column and spit out the corresponding mean encoded value 

    Args :
    -------
    Category : string = category

    Returns :
    ---------
    Mapped_category_value : int = Mean encoded value of that category
    """

    try :
        # Fetching the list of dictionary which contains the mapped mean values
        training_pipeline = client.get_pipeline("training_warden")
        last_run = training_pipeline.last_run
        dict_list = last_run.steps["categorical_encoder"].outputs["dict_list"].load()
        
        # Extracting the corresponding mean encoding value
        category_double_dict = dict_list[0]
        category_dict = category_double_dict["category"]
        mapped_category_value = category_dict[category]
        return mapped_category_value
    
    except Exception as e :

        raise e

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
    try :

        training_pipeline = client.get_pipeline("training_warden")
        last_run = training_pipeline.last_run
        reference_data = last_run.steps["model_evaluator"].outputs["reference_data"].load()
        return reference_data
    
    except Exception as e :
        
        logging.error(f"Error while fetching batch data from previous run")
        raise e

def sample_data_exporter() :
    """
    Bring last deployed model data for testing purposes

    Args :
    ------
    None

    Results :
    ---------
    sample_batch_data : csv = Exported batch data as csv in specified location
    """
    
    try :

        training_pipeline = client.get_pipeline("training_warden")
        last_run = training_pipeline.last_run
        sample_batch_data = last_run.steps["data_splitter"].outputs["batch_2"].load()
        sample_batch_data.to_csv('./data/sample_batch_data.csv', index=False)
        
        return sample_batch_data
    
    except Exception as e :
        
        logging.error(f"Error while fetching batch data from previous run")
        raise e

@step(enable_cache=False)
def notify_data_testing_results(
    results : Annotated[str, "results"],
) :
    """
    Notifies team about data test results through slackbot

    Args :
    ------
    None

    Results :
    ---------
    None
    """
    
    try :
        # Calculating number of tests passed and failed
        data = json.loads(results)
        data_tests = data["tests"]
        status = [tests['status'] for tests in data_tests]
        success_count = status.count('SUCCESS')
        failure_count = status.count('FAILURE')
        message = f"Data testing finished, Total tests : {success_count+failure_count}, Tests Passed : {success_count} ✔️, Tests Failed : {failure_count} ❌"
        warden_slackbot(message)
    
    except Exception as e :
        
        logging.error(f"Error while notifying data testing results")
        raise e
    
