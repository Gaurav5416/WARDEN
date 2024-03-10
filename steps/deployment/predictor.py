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
    ) -> Tuple[Annotated[np.ndarray, "prediction"],
               Annotated[pd.DataFrame, "batch_data"],
               ] :
    df_ytrue = df["is_fraud"]
    df.drop(columns ="is_fraud", inplace = True)
    service.start(timeout=60)
    data = df.to_numpy()
    prediction = service.predict("predict_ndarray", data)
    rich_print(prediction)
    df["y_true"] = df_ytrue
    df["y_pred"] = prediction
    return prediction, df