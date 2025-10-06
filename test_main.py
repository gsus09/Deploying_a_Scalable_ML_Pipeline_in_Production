from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_returns_expected_message():
    response = client.get("/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "Info" in response.json()
    assert (
        "ML Project Deploying_a_Scalable_ML_Pipeline_in_Production"
        in response.json()["Info"]
    )


def test_predict_income_smaler50K():
    payload = {
        "age": 10,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Preschool",
        "education_num": 0,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Unmarried",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0,
        "native_country": "Cuba",
    }
    response = client.post("/predict_income", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "Income prediction is:" in data
    assert data["Income prediction is:"] == "<=50K"


def test_predict_income_invalid_age():
    payload = {
        "age": 150,  # Invalid age
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2000,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native_country": "United-States",
    }
    response = client.post("/predict_income", json=payload)
    assert response.status_code == 422


def test_predict_income_missing_field():
    payload = {
        "age": 27,
        # "workclass" missing
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2000,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native_country": "United-States",
    }
    response = client.post("/predict_income", json=payload)
    assert response.status_code == 422


def test_predict_income_invalid_enum():
    payload = {
        "age": 27,
        "workclass": "Invalid-workclass",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2000,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native_country": "United-States",
    }
    response = client.post("/predict_income", json=payload)
    assert response.status_code == 422


def test_predict_income_greater50k():
    payload = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2000000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    response = client.post("/predict_income", json=payload)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {"Income prediction is:": ">50K"}
