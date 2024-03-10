from zenml import step
from typing import cast

from zenml.integrations.bentoml.services import BentoMLDeploymentService
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import BentoMLModelDeployer

@step(enable_cache=False)
def bentoml_prediction_service_loader(
        pipeline_name : str,
        step_name : str,
        running : bool = True,
        model_name : str = "model",
    )-> BentoMLDeploymentService:
    """Get the BentoML prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the model.
        step_name: the name of the step that deployed the model.
        model_name: the name of the model that was deployed.
    """
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running = running,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The BentoML prediction server last deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is not currently "
            f"running."
        )

    return cast(BentoMLDeploymentService, services[0])