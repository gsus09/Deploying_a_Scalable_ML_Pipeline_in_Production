import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model
from starter.ml.model import compute_model_metrics
from starter.ml.model import inference


def test_train_model_returns_trained_model():
    # Create dummy data
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])

    # Train model
    model = train_model(X_train, y_train)

    # Check that the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

    # Check that the model has been fitted (estimators_ attribute exists)
    assert hasattr(model, "estimators_")

    # Check that the model can make predictions
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_compute_model_metrics_perfect_prediction():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_compute_model_metrics_all_wrong():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 0.0
    assert recall == 0.0
    assert fbeta == 0.0


def test_compute_model_metrics_partial_correct():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    # 2 true positives, 0 false positives, 1 false negative
    assert precision == 1.0
    assert recall == 2 / 3
    assert np.isclose(fbeta, 0.8)


def test_compute_model_metrics_zero_division():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_inference_returns_predictions():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 1], [0, 0]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_inference_predictions_match_model_predict():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 1], [0, 0]])
    preds_inference = inference(model, X_test)
    preds_direct = model.predict(X_test)
    assert np.array_equal(preds_inference, preds_direct)
