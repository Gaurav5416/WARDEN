
import os

from typing import Type

import joblib
from sklearn.ensemble import AdaBoostClassifier
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "WardenEnv"


class SKLearnModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (AdaBoostClassifier, )
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[AdaBoostClassifier]) -> AdaBoostClassifier:
        """Read from artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        return joblib.load(model_path)

    def save(self, model: AdaBoostClassifier) -> None:
        """Write to artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        joblib.dump(model, model_path)