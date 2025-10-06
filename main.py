"""FastAPI app for income prediction
Author: Martin Thomas
Date: 2024-06-10
"""

from fastapi import FastAPI
from typing_extensions import Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from starter.ml.model import inference, load_artifacts, CAT_FEATURES
from starter.ml.data import process_data


app = FastAPI()


# post Input Schema
class ModelInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 27,
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
        }
    )

    age: int = Field(ge=0, lt=130, description="Age must be between 0 and 129")
    workclass: Literal[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
    ]
    fnlgt: int
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th",
    ]
    education_num: int
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ]
    occupation: Literal[
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int = Field(
        ge=0, lt=168, description="Hours per week must be between 1 and 167"
    )
    native_country: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
    ]


# Load artifacts
model, encoder, lb = load_artifacts()


# Root Path
@app.get("/")
async def root():
    return {
        "Info": (
            "Hello to ML Project Deploying_a_Scalable_ML_Pipeline_in_Production. "
            "This API checks if your salary is greater or less $50k/yr based on census data."
        )
    }


# Prediction Path
@app.post("/predict_income")
async def predict(input: ModelInput):
    input_data = np.array(
        [
            [
                input.age,
                input.workclass,
                input.fnlgt,
                input.education,
                input.education_num,
                input.marital_status,
                input.occupation,
                input.relationship,
                input.race,
                input.sex,
                input.capital_gain,
                input.capital_loss,
                input.hours_per_week,
                input.native_country,
            ]
        ]
    )

    original_cols_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

    input_df = pd.DataFrame(data=input_data, columns=original_cols_names)

    X, _, _, _ = process_data(
        input_df,
        categorical_features=CAT_FEATURES,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction is:": pred}
