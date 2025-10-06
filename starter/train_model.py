"""Train and save a machine learning model.
Authors: Martin Thomas
Date: 2024-06-10
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import (
    train_model,
    save_model,
    inference,
    compute_model_metrics,
    CAT_FEATURES,
)
from ml.performance_kpi import compute_model_performance_on_slices
from pathlib import Path
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

data_path = Path(__file__).parent.parent / "data" / "census.csv"
data = pd.read_csv(data_path)
logger.info("Data loaded successfully.")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _encoder, _lb = process_data(
    test,
    categorical_features=CAT_FEATURES,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model, encoder, lb)
logger.info("Model trained and saved successfully.")

y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logger.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

performance_df = compute_model_performance_on_slices(
    model, test, CAT_FEATURES, "salary", encoder, lb
)

performance_path = Path(__file__).parent.parent / "model" / "slice_output.csv"
performance_df.to_csv(performance_path, index=False)
logger.info(f"Model performance on slices saved to {performance_path}.")
