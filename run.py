from pipelines.training_pipeline import training_warden
from pipelines.deployment_pipeline import deploying_warden
from pipelines.inference_pipeline import inference_pipeline

from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import click


TRAIN = "train"
DEPLOY = "deploy"
PREDICT = "predict"
TRAIN_DEPLOY_PREDICT = "train_deploy_predict"

@click.command()
@click.option(
    "--config",
    type=click.Choice([TRAIN, DEPLOY, PREDICT,TRAIN_DEPLOY_PREDICT]),
    default=TRAIN_DEPLOY_PREDICT,
)


def main(config:str):

    train = config == TRAIN or config == TRAIN_DEPLOY_PREDICT
    deploy = config == DEPLOY or config == TRAIN_DEPLOY_PREDICT
    predict = config == PREDICT or config == TRAIN_DEPLOY_PREDICT
    
    if train :
        training_warden()
            
    if deploy :
        deploying_warden()
            
    if predict :
        inference_pipeline()
        
        


    
    
    print(Client().active_stack.experiment_tracker.get_tracking_uri())    
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

if __name__ == "__main__":
    main()