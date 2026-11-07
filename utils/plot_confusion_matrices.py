import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrices(
    models: dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    for name, model in models.items():
        y_pred = model.predict(X_test)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No Diabetes", "Diabetes"],
            cmap="Blues",
        )

        plt.title(name)
        plt.tight_layout()
        plt.show()