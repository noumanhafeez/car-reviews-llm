import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from linear_regression.src.data_preprocessing import preprocess


def load_model(path="artifacts/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_regression(save_path="outputs/regression_plot.png"):
    df = preprocess()

    X = df['TV'].values
    y = df['Sales'].values

    model = load_model()

    x_line = np.linspace(min(X), max(X), 100)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Data points")
    plt.plot(x_line, y_line, label="Regression Line")

    plt.xlabel("TV Advertising")
    plt.ylabel("Sales")
    plt.title("Linear Regression Fit")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Optional: still show plot
    plt.show()

    print(f"Plot saved at: {save_path}")


if __name__ == "__main__":
    plot_regression()