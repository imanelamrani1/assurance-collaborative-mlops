import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_processed_data,
    temporal_split,
    validate_feature_columns,
    validate_temporal_split
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "classification_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIRECTORY
    / "classification_metadata.json"
)

def evaluate_classifier(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> dict:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        )
    }

def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray
) -> tuple[float, float]:
    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    best_threshold = 0.50
    best_f1_score = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_f1_score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if current_f1_score > best_f1_score:
            best_f1_score = (
                current_f1_score
            )

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        float(best_f1_score)
    )

def build_candidate_pipelines() -> dict:
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor()
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor()
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight=(
                        "balanced_subsample"
                    ),
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return {
        "logistic_regression": (
            logistic_pipeline
        ),
        "random_forest": (
            random_forest_pipeline
        )
    }

def train_and_select_model(
    split_data: dict
) -> tuple[
    str,
    Pipeline,
    float,
    dict
]:
    candidates = (
        build_candidate_pipelines()
    )

    validation_results = {}
    for model_name, pipeline in (
        candidates.items()
    ):
        print(
            f"Entraînement : {model_name}"
        )

        pipeline.fit(
            split_data["X_train"],
            split_data["y_train"]
        )

        probabilities = (
            pipeline.predict_proba(
                split_data[
                    "X_validation"
                ]
            )[:, 1]
        )

        threshold, best_f1 = (
            find_best_threshold(
                y_true=split_data[
                    "y_validation"
                ],
                probabilities=probabilities
            )
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = evaluate_classifier(
            y_true=split_data[
                "y_validation"
            ],
            predictions=predictions,
            probabilities=probabilities
        )

        metrics["threshold"] = threshold
        metrics[
            "optimized_f1_score"
        ] = best_f1

        validation_results[
            model_name
        ] = metrics

    selected_model_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name][
                "pr_auc"
            ]
        )
    )

    selected_pipeline = candidates[
        selected_model_name
    ]

    selected_threshold = (
        validation_results[
            selected_model_name
        ]["threshold"]
    )

    return (
        selected_model_name,
        selected_pipeline,
        selected_threshold,
        validation_results
    )

def fit_final_model(
    data: pd.DataFrame,
    pipeline_template: Pipeline
) -> Pipeline:
    final_train_mask = data[
        "year"
    ].isin([1, 2, 3, 4])

    X_final_train = data.loc[
        final_train_mask,
        FEATURE_COLUMNS
    ].copy()

    y_final_train = data.loc[
        final_train_mask,
        CLASSIFICATION_TARGET
    ].copy()

    final_pipeline = clone(
        pipeline_template
    )

    final_pipeline.fit(
        X_final_train,
        y_final_train
    )

    return final_pipeline

def evaluate_final_model(
    data: pd.DataFrame,
    final_pipeline: Pipeline,
    threshold: float
) -> dict:
    test_mask = (
        data["year"] == 5
    )

    X_test = data.loc[
        test_mask,
        FEATURE_COLUMNS
    ].copy()

    y_test = data.loc[
        test_mask,
        CLASSIFICATION_TARGET
    ].copy()

    probabilities = (
        final_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return evaluate_classifier(
        y_true=y_test,
        predictions=predictions,
        probabilities=probabilities
    )

def save_model_and_metadata(
    pipeline: Pipeline,
    selected_model_name: str,
    selected_threshold: float,
    validation_results: dict,
    test_metrics: dict
) -> None:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )
    metadata = {
        "model_name": (
            selected_model_name
        ),
        "target": (
            CLASSIFICATION_TARGET
        ),
        "threshold": float(
            selected_threshold
        ),
        "train_years": [
            1,
            2,
            3,
            4
        ],
        "test_year": 5,
        "features": FEATURE_COLUMNS,
        "validation_results": (
            validation_results
        ),
        "test_metrics": test_metrics
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Modèle enregistré : {MODEL_PATH}"
    )

    print(
        f"Métadonnées enregistrées : "
        f"{METADATA_PATH}"
    )

def main() -> None:
    data = load_processed_data()

    validate_feature_columns(
        data
    )

    split_data = temporal_split(
        data=data,
        target_column=(
            CLASSIFICATION_TARGET
        )
    )

    validate_temporal_split(
        data=data,
        split_data=split_data
    )
    (
        selected_model_name,
        selected_pipeline,
        selected_threshold,
        validation_results
    ) = train_and_select_model(
        split_data
    )

    print(
        "Modèle sélectionné : "
        f"{selected_model_name}"
    )

    print(
        "Seuil sélectionné : "
        f"{selected_threshold:.2f}"
    )

    final_pipeline = fit_final_model(
        data=data,
        pipeline_template=(
            selected_pipeline
        )
    )

    test_metrics = evaluate_final_model(
        data=data,
        final_pipeline=final_pipeline,
        threshold=selected_threshold
    )

    print(
        "Métriques finales :"
    )

    print(
        json.dumps(
            test_metrics,
            indent=4,
            ensure_ascii=False
        )
    )

    save_model_and_metadata(
        pipeline=final_pipeline,
        selected_model_name=(
            selected_model_name
        ),
        selected_threshold=(
            selected_threshold
        ),
        validation_results=(
            validation_results
        ),
        test_metrics=test_metrics
    )

if __name__ == "__main__":
    main()
