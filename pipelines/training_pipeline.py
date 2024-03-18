from zenml import pipeline

from steps.etl.data_loader import data_loader
from steps.etl.data_balancer import data_balancer
from steps.etl.categorical_encoder import categorical_encoder
from steps.etl.feature_engineer import feature_engineer
from steps.etl.data_splitter import train_test_splitter, data_splitter
from steps.monitoring.data_tester import data_tester
from steps.monitoring.alerter import warden_slackbot
from steps.training.train_model import model_trainer
from steps.training.evaluate_model import model_evaluator
from steps.deployment.deployment_trigger import deployment_trigger
from src.utils import notify_data_testing_results, sample_data_exporter

@pipeline(enable_cache = False,)
def training_warden():
    """Train a model"""
    df = data_loader() 
    balanced_df = data_balancer(df)
    encoded_df, mapping_dict_list = categorical_encoder(balanced_df)
    transformed_df = feature_engineer(encoded_df)
    batch_1, batch_2 = data_splitter(transformed_df) 
    # sample_data_exporter() # Exports batch_2 for testing purposes
    results_json, results_html = data_tester(reference_dataset = batch_1)
    notify_data_testing_results(results_json)
    X_train, X_test, y_train, y_test = train_test_splitter(batch_1)  
    model = model_trainer(X_train, y_train)  
           
    reference_data, accuracy, precision, recall, f1 = model_evaluator(model, X_test, y_test)
    decision = deployment_trigger(evaluation_metric = f1)


