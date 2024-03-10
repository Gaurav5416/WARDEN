from zenml import pipeline
from zenml.config import DockerSettings

from zenml.integrations.constants import MLFLOW, BENTOML
from configs.deployment_config import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from steps.deployment.dynamic_importer import dynamic_importer
from steps.deployment.prediction_service_loader import bentoml_prediction_service_loader
from steps.deployment.predictor import bentoml_predictor
from steps.deployment.data_reporter import data_reporter
from steps.alerter import warden_slackbot

docker_settings = DockerSettings(required_integrations=[MLFLOW, BENTOML])
 
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline():
     
     batch_1, batch_2 = dynamic_importer()
     
     service = bentoml_prediction_service_loader(
         model_name =  MODEL_NAME,
         pipeline_name= PIPELINE_NAME,
         step_name= PIPELINE_STEP_NAME,
         running = False,
         )
     prediction, current_dataset = bentoml_predictor(service = service, df=batch_2)
     data_report  = data_reporter(reference_dataset = batch_1, comparison_dataset = current_dataset)
     warden_slackbot("✅ : Pipeline successfully ran")