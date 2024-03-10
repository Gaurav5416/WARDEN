from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    evidently_report_step,
)

data_reporter = evidently_report_step.with_options(
    
    parameters=dict(
        
        column_mapping=EvidentlyColumnMapping(
            target="y_true",
            numerical_features=["amt",
                                "age",
                                "trans_day", 
                                "trans_hour",
                                "trans_dayofweek",
                                "trans_month",
                                ],
            categorical_features=[
                "category",
                "gender_M",
                ],
            prediction="y_pred" # For checking classification preset
        ),
        
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
            EvidentlyMetricConfig.metric("TargetDriftPreset"),
            EvidentlyMetricConfig.metric("ClassificationPreset"),
        ],
    ),
)