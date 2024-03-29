from zenml import __version__ as zenml_version
from zenml.integrations.bentoml.steps import bento_builder_step

from configs.deployment_config import MODEL_NAME


bento_builder = bento_builder_step.with_options(
    parameters = dict(
        model_name = MODEL_NAME,
        model_type = "sklearn",
        service = "service.py:svc",
        labels = {
            "framework" : "sklearn",
            "dataset":"fraud",
            "zenml_version" : zenml_version,
        },
        python={
            "packages": ["zenml", "scikit-learn"],
        },
    )
)
