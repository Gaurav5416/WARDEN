from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.integrations.constants import BENTOML

docker_settings = DockerSettings(required_integrations = [MLFLOW, BENTOML])

from steps.deployment.model_loader import model_loader
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.bentoml.steps import bentoml_model_deployer_step
from steps.deployment.bento_builder import bento_builder
from configs.deployment_config import MODEL_NAME

@pipeline(enable_cache = False,
          settings={"docker": docker_settings},)

def deploying_warden() :
    """
    Continuous deployment pipeline deploys the model
    """
    # training_warden()
    model, decision = model_loader()
    bento = bento_builder(model)
    bentoml_model_deployer_step(
        bento = bento,
        deploy_decision = decision,
        model_name = MODEL_NAME,
        port = 3001,
        production = False,
        timeout = 1000,
        )