import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSIFICATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification_pipeline.joblib"
)

CLASSIFICATION_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification_metadata.json"
)

COST_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cost_model_bundle.joblib"
)

GROUPING_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "grouping_bundle.joblib"
)


model_store = {}


def read_json(
    file_path: Path,
) -> dict:
    """
    Lit un fichier JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(
            json_file
        )


def load_models() -> None:
    """
    Charge tous les modèles nécessaires
    au démarrage de l'API.
    """
    required_files = [
        CLASSIFICATION_MODEL_PATH,
        CLASSIFICATION_METADATA_PATH,
        COST_MODEL_PATH,
        GROUPING_MODEL_PATH,
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Modèles ou métadonnées absents : "
            f"{missing_files}"
        )

    model_store[
        "classification"
    ] = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    model_store[
        "classification_metadata"
    ] = read_json(
        CLASSIFICATION_METADATA_PATH
    )

    model_store[
        "cost"
    ] = joblib.load(
        COST_MODEL_PATH
    )

    model_store[
        "grouping"
    ] = joblib.load(
        GROUPING_MODEL_PATH
    )

    print(
        "Tous les modèles ont été chargés."
    )


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Gère le chargement et la libération
    des modèles pendant la vie de l'API.
    """
    load_models()

    yield

    model_store.clear()


app = FastAPI(
    title=(
        "Adaptive Collaborative "
        "Insurance API"
    ),
    description=(
        "API de prédiction du risque, "
        "du coût attendu et du segment "
        "d'un assuré."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


BinaryValue = Annotated[
    int,
    Field(
        ge=0,
        le=1,
    ),
]


class InsuranceProfile(BaseModel):
    """
    Variables nécessaires aux modèles.
    """

    gender: BinaryValue

    Age_client: Annotated[
        int,
        Field(
            ge=18,
            le=100,
        ),
    ]

    age_of_car_M: Annotated[
        int,
        Field(
            ge=0,
            le=100,
        ),
    ]

    Car_power_M: Annotated[
        float,
        Field(
            gt=0,
        ),
    ]

    Car_2ndDriver_M: BinaryValue
    num_policiesC: BinaryValue
    metro_code: BinaryValue
    Policy_PaymentMethodA: BinaryValue
    Policy_PaymentMethodH: BinaryValue

    Insuredcapital_content_re: Annotated[
        float,
        Field(
            ge=0,
        ),
    ]

    Insuredcapital_continent_re: Annotated[
        float,
        Field(
            ge=0,
        ),
    ]

    appartment: BinaryValue

    Client_Seniority: Annotated[
        float,
        Field(
            ge=0,
        ),
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": 1,
                    "Age_client": 45,
                    "age_of_car_M": 8,
                    "Car_power_M": 110.0,
                    "Car_2ndDriver_M": 0,
                    "num_policiesC": 1,
                    "metro_code": 1,
                    "Policy_PaymentMethodA": 1,
                    "Policy_PaymentMethodH": 1,
                    "Insuredcapital_content_re": 10.2,
                    "Insuredcapital_continent_re": 11.5,
                    "appartment": 1,
                    "Client_Seniority": 9.5,
                }
            ]
        }
    }


class RiskPrediction(BaseModel):
    """
    Réponse retournée par l'API.
    """

    claim_probability: float
    predicted_claim_class: int
    classification_threshold: float
    predicted_frequency: float
    predicted_severity: float
    predicted_expected_cost: float
    cost_model_approach: str
    profile_cluster: int


def profile_to_dataframe(
    profile: InsuranceProfile,
) -> pd.DataFrame:
    """
    Transforme le profil Pydantic
    en DataFrame à une ligne.
    """
    profile_dictionary = (
        profile.model_dump()
    )

    profile_dataframe = pd.DataFrame(
        [profile_dictionary]
    )

    return profile_dataframe


def predict_classification(
    profile_data: pd.DataFrame,
) -> tuple[
    float,
    int,
    float,
]:
    """
    Prédit la probabilité et la classe
    de sinistre.
    """
    classification_pipeline = (
        model_store[
            "classification"
        ]
    )

    classification_metadata = (
        model_store[
            "classification_metadata"
        ]
    )

    threshold = float(
        classification_metadata[
            "threshold"
        ]
    )

    probability = float(
        classification_pipeline
        .predict_proba(
            profile_data
        )[0, 1]
    )

    predicted_class = int(
        probability >= threshold
    )

    return (
        probability,
        predicted_class,
        threshold,
    )


def predict_cost(
    profile_data: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    str,
]:
    """
    Prédit fréquence, sévérité
    et coût attendu.
    """
    cost_bundle = model_store[
        "cost"
    ]

    approach = cost_bundle[
        "approach"
    ]

    frequency_prediction = float(
        cost_bundle[
            "frequency_pipeline"
        ].predict(
            profile_data
        )[0]
    )

    severity_prediction = float(
        cost_bundle[
            "severity_pipeline"
        ].predict(
            profile_data
        )[0]
    )

    if approach == "frequency_severity":
        expected_cost = (
            frequency_prediction
            * severity_prediction
        )
    elif approach == "tweedie":
        expected_cost = float(
            cost_bundle[
                "tweedie_pipeline"
            ].predict(
                profile_data
            )[0]
        )
    else:
        raise ValueError(
            "Approche de coût inconnue : "
            f"{approach}"
        )

    return (
        frequency_prediction,
        severity_prediction,
        float(expected_cost),
        approach,
    )


def predict_cluster(
    profile_data: pd.DataFrame,
) -> int:
    """
    Affecte le profil à son cluster.
    """
    grouping_bundle = model_store[
        "grouping"
    ]

    feature_columns = grouping_bundle[
        "features"
    ]

    scaled_profile = grouping_bundle[
        "scaler"
    ].transform(
        profile_data[
            feature_columns
        ]
    )

    cluster_label = int(
        grouping_bundle[
            "kmeans_model"
        ].predict(
            scaled_profile
        )[0]
        + 1
    )

    return cluster_label


@app.get(
    "/",
    tags=["General"],
)
def root() -> dict:
    """
    Présente brièvement l'API.
    """
    return {
        "message": (
            "Adaptive Collaborative "
            "Insurance API"
        ),
        "documentation": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get(
    "/health",
    tags=["General"],
)
def health_check() -> dict:
    """
    Vérifie que les modèles sont chargés.
    """
    required_models = {
        "classification",
        "classification_metadata",
        "cost",
        "grouping",
    }

    loaded_models = set(
        model_store.keys()
    )

    models_ready = (
        required_models
        .issubset(
            loaded_models
        )
    )

    return {
        "status": (
            "healthy"
            if models_ready
            else "unhealthy"
        ),
        "models_loaded": models_ready,
    }


@app.post(
    "/predict",
    response_model=RiskPrediction,
    tags=["Prediction"],
)
def predict_risk(
    profile: InsuranceProfile,
) -> RiskPrediction:
    """
    Produit toutes les prédictions
    nécessaires à la gestion du risque.
    """
    try:
        profile_data = (
            profile_to_dataframe(
                profile
            )
        )

        (
            probability,
            predicted_class,
            threshold,
        ) = predict_classification(
            profile_data
        )

        (
            frequency,
            severity,
            expected_cost,
            cost_approach,
        ) = predict_cost(
            profile_data
        )

        cluster_label = predict_cluster(
            profile_data
        )

        return RiskPrediction(
            claim_probability=probability,
            predicted_claim_class=(
                predicted_class
            ),
            classification_threshold=(
                threshold
            ),
            predicted_frequency=frequency,
            predicted_severity=severity,
            predicted_expected_cost=(
                expected_cost
            ),
            cost_model_approach=(
                cost_approach
            ),
            profile_cluster=(
                cluster_label
            ),
        )

    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant la prédiction : "
                f"{exception}"
            ),
        ) from exception
