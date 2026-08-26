import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import (
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    COST_TARGET,
    FEATURE_COLUMNS,
    FREQUENCY_TARGET,
    SEVERITY_TARGET,
    build_preprocessor,
    load_processed_data,
    validate_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY = PROJECT_ROOT / "models"

CLASSIFICATION_MODEL_PATH = (
    MODEL_DIRECTORY
    / "classification_pipeline.joblib"
)

COST_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cost_model_bundle.joblib"
)

COST_METADATA_PATH = (
    MODEL_DIRECTORY
    / "cost_model_metadata.json"
)

RISK_SCORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_scores_year5.csv"
)


def evaluate_cost_model(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """
    Évalue une estimation du coût total.
    """
    safe_predictions = np.clip(
        predictions,
        1e-10,
        None,
    )

    real_total = float(
        y_true.sum()
    )

    predicted_total = float(
        predictions.sum()
    )

    if real_total == 0:
        total_error_percentage = None
    else:
        total_error_percentage = float(
            (
                predicted_total
                - real_total
            )
            / real_total
            * 100
        )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions,
                )
            )
        ),
        "tweedie_deviance": float(
            mean_tweedie_deviance(
                y_true,
                safe_predictions,
                power=1.5,
            )
        ),
        "real_mean": float(
            y_true.mean()
        ),
        "predicted_mean": float(
            predictions.mean()
        ),
        "real_total": real_total,
        "predicted_total": predicted_total,
        "total_error_percentage": (
            total_error_percentage
        ),
    }


def build_frequency_pipeline() -> Pipeline:
    """
    Construit le pipeline de fréquence.
    """
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "model",
                PoissonRegressor(
                    alpha=1.0,
                    max_iter=1000,
                ),
            ),
        ]
    )


def build_severity_pipeline() -> Pipeline:
    """
    Construit le pipeline de sévérité.
    """
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "model",
                GammaRegressor(
                    alpha=1.0,
                    max_iter=1000,
                ),
            ),
        ]
    )


def build_tweedie_pipeline() -> Pipeline:
    """
    Construit le pipeline de coût Tweedie.
    """
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "model",
                TweedieRegressor(
                    power=1.5,
                    alpha=1.0,
                    link="log",
                    max_iter=1000,
                ),
            ),
        ]
    )


def train_validation_models(
    data: pd.DataFrame,
) -> tuple[
    str,
    dict,
    dict,
]:
    """
    Entraîne les modèles sur les années 1 à 3
    et les compare sur l'année 4.
    """
    train_mask = data["year"].isin(
        [1, 2, 3]
    )

    validation_mask = (
        data["year"] == 4
    )

    severity_train_mask = (
        train_mask
        & (data[COST_TARGET] > 0)
    )

    X_train = data.loc[
        train_mask,
        FEATURE_COLUMNS,
    ].copy()

    X_validation = data.loc[
        validation_mask,
        FEATURE_COLUMNS,
    ].copy()

    y_train_frequency = data.loc[
        train_mask,
        FREQUENCY_TARGET,
    ].copy()

    y_train_total_cost = data.loc[
        train_mask,
        COST_TARGET,
    ].copy()

    y_validation_total_cost = data.loc[
        validation_mask,
        COST_TARGET,
    ].copy()

    X_train_severity = data.loc[
        severity_train_mask,
        FEATURE_COLUMNS,
    ].copy()

    y_train_severity = data.loc[
        severity_train_mask,
        SEVERITY_TARGET,
    ].copy()

    if len(X_train_severity) == 0:
        raise ValueError(
            "Aucune observation positive "
            "pour entraîner la sévérité."
        )

    frequency_pipeline = (
        build_frequency_pipeline()
    )

    severity_pipeline = (
        build_severity_pipeline()
    )

    tweedie_pipeline = (
        build_tweedie_pipeline()
    )

    print(
        "Entraînement de la fréquence..."
    )

    frequency_pipeline.fit(
        X_train,
        y_train_frequency,
    )

    print(
        "Entraînement de la sévérité..."
    )

    severity_pipeline.fit(
        X_train_severity,
        y_train_severity,
    )

    print(
        "Entraînement du modèle Tweedie..."
    )

    tweedie_pipeline.fit(
        X_train,
        y_train_total_cost,
    )

    frequency_predictions = (
        frequency_pipeline.predict(
            X_validation
        )
    )

    severity_predictions = (
        severity_pipeline.predict(
            X_validation
        )
    )

    frequency_severity_predictions = (
        frequency_predictions
        * severity_predictions
    )

    tweedie_predictions = (
        tweedie_pipeline.predict(
            X_validation
        )
    )

    validation_results = {
        "frequency_severity": (
            evaluate_cost_model(
                y_true=(
                    y_validation_total_cost
                ),
                predictions=(
                    frequency_severity_predictions
                ),
            )
        ),
        "tweedie": (
            evaluate_cost_model(
                y_true=(
                    y_validation_total_cost
                ),
                predictions=(
                    tweedie_predictions
                ),
            )
        ),
    }

    selected_approach = min(
        validation_results,
        key=lambda name: (
            validation_results[name][
                "tweedie_deviance"
            ]
        ),
    )

    validation_pipelines = {
        "frequency_pipeline": (
            frequency_pipeline
        ),
        "severity_pipeline": (
            severity_pipeline
        ),
        "tweedie_pipeline": (
            tweedie_pipeline
        ),
    }

    return (
        selected_approach,
        validation_results,
        validation_pipelines,
    )


def train_final_models(
    data: pd.DataFrame,
) -> dict:
    """
    Réentraîne tous les modèles de coût
    sur les années 1 à 4.
    """
    final_train_mask = data[
        "year"
    ].isin([1, 2, 3, 4])

    final_severity_mask = (
        final_train_mask
        & (data[COST_TARGET] > 0)
    )

    X_final_train = data.loc[
        final_train_mask,
        FEATURE_COLUMNS,
    ].copy()

    y_final_frequency = data.loc[
        final_train_mask,
        FREQUENCY_TARGET,
    ].copy()

    y_final_total_cost = data.loc[
        final_train_mask,
        COST_TARGET,
    ].copy()

    X_final_severity = data.loc[
        final_severity_mask,
        FEATURE_COLUMNS,
    ].copy()

    y_final_severity = data.loc[
        final_severity_mask,
        SEVERITY_TARGET,
    ].copy()

    final_frequency_pipeline = clone(
        build_frequency_pipeline()
    )

    final_severity_pipeline = clone(
        build_severity_pipeline()
    )

    final_tweedie_pipeline = clone(
        build_tweedie_pipeline()
    )

    print(
        "Réentraînement final de la fréquence..."
    )

    final_frequency_pipeline.fit(
        X_final_train,
        y_final_frequency,
    )

    print(
        "Réentraînement final de la sévérité..."
    )

    final_severity_pipeline.fit(
        X_final_severity,
        y_final_severity,
    )

    print(
        "Réentraînement final Tweedie..."
    )

    final_tweedie_pipeline.fit(
        X_final_train,
        y_final_total_cost,
    )

    return {
        "frequency_pipeline": (
            final_frequency_pipeline
        ),
        "severity_pipeline": (
            final_severity_pipeline
        ),
        "tweedie_pipeline": (
            final_tweedie_pipeline
        ),
    }


def predict_final_costs(
    data: pd.DataFrame,
    final_models: dict,
    selected_approach: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Produit les scores de risque
    pour l'année 5.
    """
    test_mask = (
        data["year"] == 5
    )

    X_test = data.loc[
        test_mask,
        FEATURE_COLUMNS,
    ].copy()

    y_test_total_cost = data.loc[
        test_mask,
        COST_TARGET,
    ].copy()

    frequency_predictions = (
        final_models[
            "frequency_pipeline"
        ].predict(X_test)
    )

    severity_predictions = (
        final_models[
            "severity_pipeline"
        ].predict(X_test)
    )

    frequency_severity_predictions = (
        frequency_predictions
        * severity_predictions
    )

    tweedie_predictions = (
        final_models[
            "tweedie_pipeline"
        ].predict(X_test)
    )

    if selected_approach == (
        "frequency_severity"
    ):
        expected_cost_predictions = (
            frequency_severity_predictions
        )
    else:
        expected_cost_predictions = (
            tweedie_predictions
        )

    test_results = {
        "frequency_severity": (
            evaluate_cost_model(
                y_true=y_test_total_cost,
                predictions=(
                    frequency_severity_predictions
                ),
            )
        ),
        "tweedie": (
            evaluate_cost_model(
                y_true=y_test_total_cost,
                predictions=(
                    tweedie_predictions
                ),
            )
        ),
        "selected_approach": (
            selected_approach
        ),
    }

    if not (
        CLASSIFICATION_MODEL_PATH.exists()
    ):
        raise FileNotFoundError(
            "Le modèle de classification "
            "est absent. Exécutez d'abord "
            "train_classification.py."
        )

    classification_pipeline = (
        joblib.load(
            CLASSIFICATION_MODEL_PATH
        )
    )

    claim_probabilities = (
        classification_pipeline
        .predict_proba(X_test)[:, 1]
    )

    risk_scores = data.loc[
        test_mask,
        [
            "PolID",
            "year",
            "Has_Claim",
            "Total_NClaims",
            "Total_Claims",
        ],
    ].copy()

    risk_scores[
        "Predicted_Claim_Probability"
    ] = claim_probabilities

    risk_scores[
        "Predicted_Frequency"
    ] = frequency_predictions

    risk_scores[
        "Predicted_Severity"
    ] = severity_predictions

    risk_scores[
        "Predicted_Expected_Cost"
    ] = expected_cost_predictions

    risk_scores["Risk_Decile"] = (
        pd.qcut(
            risk_scores[
                "Predicted_Expected_Cost"
            ],
            q=10,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    risk_scores["Risk_Level"] = pd.cut(
        risk_scores["Risk_Decile"],
        bins=[0, 4, 7, 10],
        labels=[
            "Faible",
            "Moyen",
            "Élevé",
        ],
        include_lowest=True,
    )

    return (
        risk_scores,
        test_results,
    )


def save_outputs(
    selected_approach: str,
    validation_results: dict,
    final_models: dict,
    risk_scores: pd.DataFrame,
    test_results: dict,
) -> None:
    """
    Enregistre les modèles, métadonnées
    et scores de risque.
    """
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RISK_SCORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "approach": selected_approach,
        "frequency_pipeline": (
            final_models[
                "frequency_pipeline"
            ]
        ),
        "severity_pipeline": (
            final_models[
                "severity_pipeline"
            ]
        ),
        "tweedie_pipeline": (
            final_models[
                "tweedie_pipeline"
            ]
        ),
    }

    joblib.dump(
        model_bundle,
        COST_MODEL_PATH,
    )

    metadata = {
        "selected_approach": (
            selected_approach
        ),
        "selection_metric": (
            "tweedie_deviance"
        ),
        "train_years": [
            1,
            2,
            3,
            4,
        ],
        "validation_year": 4,
        "test_year": 5,
        "features": FEATURE_COLUMNS,
        "validation_results": (
            validation_results
        ),
        "test_results": test_results,
    }

    with open(
        COST_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False,
        )

    risk_scores.to_csv(
        RISK_SCORE_PATH,
        index=False,
    )

    print(
        f"Modèles enregistrés : "
        f"{COST_MODEL_PATH}"
    )

    print(
        f"Métadonnées enregistrées : "
        f"{COST_METADATA_PATH}"
    )

    print(
        f"Scores enregistrés : "
        f"{RISK_SCORE_PATH}"
    )


def main() -> None:
    """
    Exécute le pipeline complet
    de modélisation du coût.
    """
    print(
        "Chargement des données..."
    )

    data = load_processed_data()

    validate_feature_columns(
        data
    )

    (
        selected_approach,
        validation_results,
        _,
    ) = train_validation_models(
        data
    )

    print(
        "Approche sélectionnée : "
        f"{selected_approach}"
    )

    print(
        "Résultats de validation :"
    )

    print(
        json.dumps(
            validation_results,
            indent=4,
            ensure_ascii=False,
        )
    )

    final_models = train_final_models(
        data
    )

    (
        risk_scores,
        test_results,
    ) = predict_final_costs(
        data=data,
        final_models=final_models,
        selected_approach=(
            selected_approach
        ),
    )

    print(
        "Résultats finaux :"
    )

    print(
        json.dumps(
            test_results,
            indent=4,
            ensure_ascii=False,
        )
    )

    save_outputs(
        selected_approach=(
            selected_approach
        ),
        validation_results=(
            validation_results
        ),
        final_models=final_models,
        risk_scores=risk_scores,
        test_results=test_results,
    )


if __name__ == "__main__":
    main()
