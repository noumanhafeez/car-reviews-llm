from sklearn.linear_model import LinearRegression
import numpy as np
from linear_regression.utils.logger import get_logger

logger = get_logger("sklearn_model", "logs/sklearn_model.log")


class SklearnLinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        logger.info("Initialized Sklearn LinearRegression")

    def fit(self, X, y):
        try:
            X = np.array(X).reshape(-1, 1)
            y = np.array(y)

            self.model.fit(X, y)
            logger.info("Sklearn model training completed")

        except Exception as e:
            logger.exception(f"Sklearn training failed: {e}")
            raise

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return self.model.predict(X)

    def get_params(self):
        return self.model.coef_[0], self.model.intercept_