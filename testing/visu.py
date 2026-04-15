import pickle
import numpy as np
from linear_regression.utils.logger import get_logger

logger = get_logger("predict", "logs/predict.log")


def load_model(path="artifacts/model.pkl"):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise


def predict(input_values):
    try:
        model = load_model()

        preds = model.predict(np.array(input_values))

        return preds

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    sample = [230.1, 151.5, 180.8]
    print(predict(sample))