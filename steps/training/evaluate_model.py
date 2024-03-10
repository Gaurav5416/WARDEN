import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from sklearn.ensemble import AdaBoostClassifier

from src.training.model_evaluation import Accuracy, Precision, Recall, F1Score

@step(enable_cache = False)
def model_evaluator(
    model: AdaBoostClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
) -> Tuple[Annotated[pd.DataFrame, "batch_data"],
           Annotated[float, "f1"],
           ] :
        
        """
        Args:
        -------
        model: AdaBoostClassifier
        x_test: pd.DataFrame
        y_test: pd.Series
        
        Returns:
        ---------
        batch_data : dataframe
        f1 : float
        
        """
        
        try:
            prediction = model.predict(X_test)
            

            accuracy = Accuracy()
            accuracy = accuracy.calculate_score(y_test, prediction)
            
            precision = Precision()
            precision = precision.calculate_score(y_test, prediction)
            
            recall = Recall()
            recall = recall.calculate_score(y_test, prediction)
            
            f1 = F1Score()
            f1 = f1.calculate_score(y_test, prediction)

            batch_data = X_test
            batch_data["y_true"] = y_test
            batch_data["y_pred"] = prediction
            
            return batch_data, f1
        
        except Exception as e:
             logging.error(e)
             raise e
