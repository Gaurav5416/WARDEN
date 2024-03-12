zenml integration install evidently

zenml integration install bentoml

zenml integration install mlflow 

zenml integration install slack


zenml data-validator register evidently_validator --flavor=evidently

zenml model-deployer register bentoml_deployer --flavor=bentoml

zenml experiment-tracker register mlflow_tracker --flavor=mlflow

zenml alerter register slack_alerter --flavor=slack --slack_token=<> --default_slack_channel_id=<>

zenml stack register warden_stack -d bentoml_deployer -e mlflow_tracker -al slack_alerter -dv evidently_validator -o default -a default

zenml stack set warden_stack