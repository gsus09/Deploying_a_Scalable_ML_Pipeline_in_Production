"""Module for computing model performance on data slices.
Authors: Martin Thomas
Date: 2024-06-10
"""

import pandas as pd
import logging
import sys

from ml.data import process_data
from ml.model import compute_model_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def compute_model_performance_on_slices(
    model, data: pd.DataFrame, categorical_features: list, label: str, encoder, lb
) -> pd.DataFrame:
    """
    Compute model performance on slices of the data based on categorical features.

    Inputs
    ------
    model : Trained machine learning model
        The model to evaluate.
    data : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features : list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing performance metrics for each slice of data.
    """
    results = []

    for feature in categorical_features:
        unique_values = data[feature].unique()
        for value in unique_values:
            slice_data = data[data[feature] == value]
            if slice_data.empty:
                continue

            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            if len(y_slice) == 0:
                continue

            y_pred = model.predict(X_slice)

            precision, recall, fbeta = compute_model_metrics(y_slice, y_pred)

            results.append(
                {
                    "feature": feature,
                    "value": value,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                    "num_samples": len(y_slice),
                }
            )

    results_df = pd.DataFrame(results)
    return results_df
