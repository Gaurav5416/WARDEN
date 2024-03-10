import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass



class Accuracy(Evaluation):
    """
    Evaluation strategy that uses Accuracy
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            accuracy: float
        """
        try:
            logging.info("Entered the calculate_score method of the Accuracy class")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("The accuracy score : " + str(round(accuracy, 4)))
            return round(accuracy, 4)
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Accuracy class. Exception message:  "
                + str(e)
            )
            raise e


class Precision(Evaluation):
    """
    Evaluation strategy that uses Precision
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            precision: float
        """
        try:
            logging.info("Entered the calculate_score method of the Precision class")
            precision = precision_score(y_true, y_pred)
            logging.info("The precision score : " + str(round(precision, 4)))
            return round(precision, 4)
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the Precision class. Exception message:  "
                + str(e)
            )
            raise e


class Recall(Evaluation):
    """
    Evaluation strategy that uses Recall
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            recall: float
        """
        try:

            logging.info("Entered the calculate_score method of the Recall class")
            recall = recall_score(y_true, y_pred)
            logging.info("The recall score : " + str(round(recall, 4)))
            return round(recall, 4)
        
        except Exception as e:

            logging.error(
                "Exception occurred in calculate_score method of the Recall class. Exception message:  "
                + str(e)
            )
            raise e

class F1Score(Evaluation):
    """
    Evaluation strategy that uses F1 Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            f1_score: float
        """
        try:

            logging.info("Entered the calculate_score method of the F1Score class")
            f1 = f1_score(y_true, y_pred)
            logging.info("The F1 score : " + str(round(f1, 4)))
            return round(f1, 4)
        
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the F1Score class. Exception message:  "
                + str(e)
            )
            raise e