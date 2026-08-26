from fastapi.testclient import TestClient

from src.api.app import app


VALID_PROFILE = {
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


def test_root_endpoint() -> None:
    """
    Vérifie que la racine de l'API
    répond correctement.
    """
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    response_data = response.json()

    assert "message" in response_data
    assert response_data[
        "documentation"
    ] == "/docs"


def test_health_endpoint() -> None:
    """
    Vérifie que tous les modèles
    sont chargés.
    """
    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data[
        "status"
    ] == "healthy"

    assert response_data[
        "models_loaded"
    ] is True


def test_prediction_endpoint() -> None:
    """
    Vérifie qu'un profil valide produit
    toutes les prédictions attendues.
    """
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=VALID_PROFILE,
        )

    assert response.status_code == 200

    prediction = response.json()

    required_fields = {
        "claim_probability",
        "predicted_claim_class",
        "classification_threshold",
        "predicted_frequency",
        "predicted_severity",
        "predicted_expected_cost",
        "cost_model_approach",
        "profile_cluster",
    }

    assert required_fields.issubset(
        prediction.keys()
    )

    assert (
        0
        <= prediction[
            "claim_probability"
        ]
        <= 1
    )

    assert prediction[
        "predicted_claim_class"
    ] in [0, 1]

    assert (
        0
        <= prediction[
            "classification_threshold"
        ]
        <= 1
    )

    assert prediction[
        "predicted_frequency"
    ] > 0

    assert prediction[
        "predicted_severity"
    ] > 0

    assert prediction[
        "predicted_expected_cost"
    ] > 0

    assert prediction[
        "profile_cluster"
    ] >= 1

    assert prediction[
        "cost_model_approach"
    ] in [
        "frequency_severity",
        "tweedie",
    ]


def test_invalid_binary_value() -> None:
    """
    Vérifie qu'une valeur binaire invalide
    est refusée par Pydantic.
    """
    invalid_profile = (
        VALID_PROFILE.copy()
    )

    invalid_profile["gender"] = 3

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_profile,
        )

    assert response.status_code == 422


def test_invalid_age() -> None:
    """
    Vérifie qu'un âge impossible
    est refusé.
    """
    invalid_profile = (
        VALID_PROFILE.copy()
    )

    invalid_profile[
        "Age_client"
    ] = 10

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_profile,
        )

    assert response.status_code == 422


def test_missing_required_feature() -> None:
    """
    Vérifie qu'une variable absente
    provoque une erreur de validation.
    """
    incomplete_profile = (
        VALID_PROFILE.copy()
    )

    incomplete_profile.pop(
        "Car_power_M"
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=incomplete_profile,
        )

    assert response.status_code == 422
