from zenml import step
from typing_extensions import Annotated
from configs.deployment_config import min_accuracy

@step(enable_cache = False)
def deployment_trigger(
    evaluation_metric : float,
    min_accuracy : float = min_accuracy,
) -> Annotated[bool, "decision"]:
    
    """
    Implements a simple model deployment trigger that looks at the
    input model evaluation metric and decides if it is good enough to deploy
    
    Parameters :
    -------------
        evaluation_metric = An eval metric as set in deployment cofig, default is f1
        min_accuracy = Minimum accuracy required to deploy the model

    Results :
    ----------
        decision = Boolean value of whether the model should be deployed or not
    """
    decision = evaluation_metric > min_accuracy
    return decision