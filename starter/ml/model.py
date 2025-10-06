"""Helping functions for train and save a machine learning model.
Authors: Martin Thomas
Date: 2024-06-10
"""

import joblib
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(
    model,
    encoder,
    lb,
    model_path=Path(__file__).parent.parent.parent / "model" / "model.pkl",
    encoder_path=Path(__file__).parent.parent.parent / "model" / "encoder.pkl",
    lb_path=Path(__file__).parent.parent.parent / "model" / "lb.pkl",
):
    """Save the trained model, encoder, and label binarizer to disk.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    model_path : str
        Path to save the model (default="model/model.pkl").
    encoder_path : str
        Path to save the encoder (default="model/encoder.pkl").
    lb_path : str
        Path to save the label binarizer (default="model/lb.pkl").
    """

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)


def load_artifacts(
    model_path=Path(__file__).parent.parent.parent / "model" / "model.pkl",
    encoder_path=Path(__file__).parent.parent.parent / "model" / "encoder.pkl",
    lb_path=Path(__file__).parent.parent.parent / "model" / "lb.pkl",
):
    """Load the trained model, encoder, and label binarizer from disk.

    Inputs
    ------
    model_path : str
        Path to load the model (default="model/model.pkl").
    encoder_path : str
        Path to load the encoder (default="model/encoder.pkl").
    lb_path : str
        Path to load the label binarizer (default="model/lb.pkl").
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    """

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb
