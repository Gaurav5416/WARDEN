from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    evidently_test_step,
)
from zenml.integrations.evidently.tests import EvidentlyTestConfig
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig

data_tester = evidently_test_step.with_options(
    
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

        tests=[
            EvidentlyTestConfig.test("DataQualityTestPreset"),
            # EvidentlyTestConfig.test("NoTargetPerformanceTestPreset"),
            # EvidentlyTestConfig.test("DataStabilityTestPreset"),
            # EvidentlyTestConfig.test("DataDriftTestPreset"),
            # EvidentlyTestConfig.test("BinaryClassificationTestPreset"),
        ],

    ),
)



