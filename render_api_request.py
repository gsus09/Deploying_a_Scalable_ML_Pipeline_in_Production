import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

features = {
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

app_url = "https://deploying-a-scalable-ml-pipeline-in-pcgr.onrender.com/predict_income"

r = requests.post(app_url, json=features)
assert r.status_code == 200

logger.info("Testing Render app")
logger.info(f"Status code: {r.status_code}")
logger.info(f"Response body: {r.json()}")
