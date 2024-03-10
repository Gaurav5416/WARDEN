from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import ClassifierMixin
from zenml import step, Model
from typing_extensions import Annotated
from zenml.client import Client
from typing import Tuple

 
@step(enable_cache=False)
def model_loader() -> Tuple[
    Annotated[AdaBoostClassifier, "trained_model"],
    Annotated[bool, "decision"]] :
    
    """Implements a simple model loader that loads the current production model
    Invokes the deployment decision from previous run
    """
    client = Client()
    training_pipeline = client.get_pipeline("training_warden")
    last_run = training_pipeline.last_run
    model = last_run.steps["model_trainer"].output.load()
    decision = last_run.steps["deployment_trigger"].output.load()
    return model, decision