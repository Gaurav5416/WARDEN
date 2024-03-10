import json

import numpy as np
import pandas as pd
from zenml import step
from rich import print as rich_print
from typing import Tuple 
from typing_extensions import Annotated
from zenml.integrations.bentoml.services import BentoMLDeploymentService

@step(enable_cache=False)
def bentoml_predictor(
    service : BentoMLDeploymentService,
    df:pd.DataFrame,
    ) -> Tuple[Annotated[pd.DataFrame, "prepared_data"],
               Annotated[pd.DataFrame, "filtered_data"],] :
    
    

    y_true = df["is_fraud"]
    df.drop(columns ="is_fraud", inplace = True)
    service.start(timeout=60)
    data = df.to_numpy()
    prediction = service.predict("predict_ndarray", data)
    # rich_print(prediction)
    df["y_true"] = y_true
    df["y_pred"] = prediction
    filtered_df = df[df['y_pred'] == 1]
    # Drop the 'prediction' column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['y_pred', 'y_true'])
    
    return df, filtered_df