import logging
import pandas as pd
from typing_extensions import Annotated

from zenml import step, ArtifactConfig
from zenml.client import Client

import mlflow
import mlflow.sklearn
from materializer.custom_materializer import SKLearnModelMaterializer
from src.training.model_building import Adaboost
from sklearn.ensemble import AdaBoostClassifier

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache = False, 
      experiment_tracker=experiment_tracker.name, 
      output_materializers={"trained_model": SKLearnModelMaterializer})
def model_trainer(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"],
) -> Annotated[
      AdaBoostClassifier, 
      ArtifactConfig(name="trained_model", is_model_artifact=True)]:
        
        """Trains a adaptive boosting ensemble model and outputs the model artifact."""
        
        try:
            
            mlflow.sklearn.autolog()
            adaboost = Adaboost(X_train, y_train)
            trained_model = adaboost.train()
            logging.info("Model trained successfully")
            return trained_model
        
        except Exception as e:
             
             logging.error(e)
             raise e

