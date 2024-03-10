from zenml import pipeline
from configs.etl_config import table_name

from steps.etl.data_loader import data_loader
from steps.etl.data_balancer import data_balancer
from steps.etl.categorical_encoder import categorical_encoder
from steps.etl.feature_engineer import feature_engineer

from steps.deployment.data_reporter import data_reporter
from steps.training.data_tester import data_tester
from steps.training.data_splitter import train_test_splitter, data_splitter
from steps.training.train_model import model_trainer
from steps.training.evaluate_model import model_evaluator

from steps.deployment.deployment_trigger import deployment_trigger
from src.utils import get_reference_data

@pipeline(enable_cache = False,)
def training_warden():
    """Train a model"""
    df = data_loader(table_name) 
    balanced_df = data_balancer(df)
    encoded_df, mapping_dict_list = categorical_encoder(balanced_df)
    transformed_df = feature_engineer(encoded_df)

    batch_1, batch_2 = data_splitter(transformed_df) # Divide data into two equal parts

    X_train, X_test, y_train, y_test = train_test_splitter(batch_1)  
    model = model_trainer(X_train, y_train)         
    batch_1, f1 = model_evaluator(model, X_test, y_test)

    #Comment below 2 lines if running the model for first time
    reference_data = get_reference_data()
    test_results = data_tester(reference_dataset = reference_data, comparison_dataset = batch_1)
    #-----------------------------------------------------
    decision = deployment_trigger(evaluation_metric = f1)


