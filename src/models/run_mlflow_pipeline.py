import json
from pathlib import Path
from typing import Any

import mlflow

from src.data.prepare_data import (
    main as prepare_data_main,
)
from src.models.build_groups import (
    main as build_groups_main,
)
from src.models.train_classification import (
    main as train_classification_main,
)
from src.models.train_cost import (
    main as train_cost_main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DATABASE_PATH = (
    PROJECT_ROOT
    / "mlflow.db"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CLASSIFICATION_METADATA_PATH = (
    MODEL_DIRECTORY
    / "classification_metadata.json"
)

CLASSIFICATION_MODEL_PATH = (
    MODEL_DIRECTORY
    / "classification_pipeline.joblib"
)

COST_METADATA_PATH = (
    MODEL_DIRECTORY
    / "cost_model_metadata.json"
)

COST_MODEL_PATH = (
    MODEL_DIRECTORY
    / "cost_model_bundle.joblib"
)

GROUPING_METADATA_PATH = (
    MODEL_DIRECTORY
    / "grouping_metadata.json"
)

GROUPING_MODEL_PATH = (
    MODEL_DIRECTORY
    / "grouping_bundle.joblib"
)

RISK_SCORE_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "risk_scores_year5.csv"
)

GROUP_ASSIGNMENTS_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "collaborative_group_assignments.csv"
)

GROUP_SUMMARY_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "collaborative_group_summary.csv"
)

EXPERIMENT_NAME = (
    "adaptive-collaborative-insurance"
)


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


def log_numeric_metrics(
    metrics: dict,
    prefix: str = "",
) -> None:
    """
    Enregistre récursivement toutes les
    valeurs numériques d'un dictionnaire.
    """
    for key, value in metrics.items():
        metric_name = (
            f"{prefix}_{key}"
            if prefix
            else key
        )

        if isinstance(
            value,
            bool,
        ):
            continue

        if isinstance(
            value,
            (int, float),
        ):
            if value is not None:
                mlflow.log_metric(
                    metric_name,
                    float(value),
                )

        elif isinstance(
            value,
            dict,
        ):
            log_numeric_metrics(
                metrics=value,
                prefix=metric_name,
            )


def log_existing_artifact(
    file_path: Path,
    artifact_path: str,
) -> None:
    """
    Enregistre un fichier seulement
    s'il existe.
    """
    if file_path.exists():
        mlflow.log_artifact(
            str(file_path),
            artifact_path=artifact_path,
        )
    else:
        print(
            "Artefact absent, non enregistré : "
            f"{file_path}"
        )


def run_data_preparation() -> None:
    """
    Exécute et trace la préparation
    des données.
    """
    with mlflow.start_run(
        run_name="data_preparation",
        nested=True,
    ):
        prepare_data_main()

        cleaned_data_path = (
            PROCESSED_DATA_DIRECTORY
            / "cleaned_data.csv"
        )

        mlflow.log_param(
            "input_file",
            "data/raw/data_ex.csv",
        )

        mlflow.log_param(
            "output_file",
            "data/processed/cleaned_data.csv",
        )

        log_existing_artifact(
            file_path=cleaned_data_path,
            artifact_path="data",
        )


def run_classification_training() -> None:
    """
    Exécute et trace la classification.
    """
    with mlflow.start_run(
        run_name="classification",
        nested=True,
    ):
        train_classification_main()

        metadata = read_json(
            CLASSIFICATION_METADATA_PATH
        )

        mlflow.log_param(
            "model_name",
            metadata["model_name"],
        )

        mlflow.log_param(
            "target",
            metadata["target"],
        )

        mlflow.log_param(
            "threshold",
            metadata["threshold"],
        )

        mlflow.log_param(
            "number_of_features",
            len(metadata["features"]),
        )

        mlflow.log_param(
            "train_years",
            str(metadata["train_years"]),
        )

        mlflow.log_param(
            "test_year",
            metadata["test_year"],
        )

        log_numeric_metrics(
            metrics=metadata[
                "test_metrics"
            ],
            prefix="test",
        )

        validation_results = metadata[
            "validation_results"
        ]

        for model_name, metrics in (
            validation_results.items()
        ):
            log_numeric_metrics(
                metrics=metrics,
                prefix=(
                    f"validation_{model_name}"
                ),
            )

        log_existing_artifact(
            file_path=(
                CLASSIFICATION_MODEL_PATH
            ),
            artifact_path="models",
        )

        log_existing_artifact(
            file_path=(
                CLASSIFICATION_METADATA_PATH
            ),
            artifact_path="metadata",
        )


def run_cost_training() -> None:
    """
    Exécute et trace les modèles de coût.
    """
    with mlflow.start_run(
        run_name="frequency_severity_cost",
        nested=True,
    ):
        train_cost_main()

        metadata = read_json(
            COST_METADATA_PATH
        )

        selected_approach = metadata[
            "selected_approach"
        ]

        mlflow.log_param(
            "selected_approach",
            selected_approach,
        )

        mlflow.log_param(
            "selection_metric",
            metadata["selection_metric"],
        )

        mlflow.log_param(
            "number_of_features",
            len(metadata["features"]),
        )

        mlflow.log_param(
            "train_years",
            str(metadata["train_years"]),
        )

        validation_results = metadata[
            "validation_results"
        ]

        for approach_name, metrics in (
            validation_results.items()
        ):
            log_numeric_metrics(
                metrics=metrics,
                prefix=(
                    f"validation_"
                    f"{approach_name}"
                ),
            )

        test_results = metadata[
            "test_results"
        ]

        for approach_name, metrics in (
            test_results.items()
        ):
            if isinstance(
                metrics,
                dict,
            ):
                log_numeric_metrics(
                    metrics=metrics,
                    prefix=(
                        f"test_{approach_name}"
                    ),
                )

        log_existing_artifact(
            file_path=COST_MODEL_PATH,
            artifact_path="models",
        )

        log_existing_artifact(
            file_path=COST_METADATA_PATH,
            artifact_path="metadata",
        )

        log_existing_artifact(
            file_path=RISK_SCORE_PATH,
            artifact_path="predictions",
        )


def run_group_construction() -> None:
    """
    Exécute et trace le clustering et
    la construction des groupes.
    """
    with mlflow.start_run(
        run_name="collaborative_groups",
        nested=True,
    ):
        build_groups_main()

        metadata = read_json(
            GROUPING_METADATA_PATH
        )

        mlflow.log_param(
            "year",
            metadata["year"],
        )

        mlflow.log_param(
            "number_of_features",
            len(metadata["features"]),
        )

        mlflow.log_param(
            "number_of_clusters",
            metadata[
                "best_number_of_clusters"
            ],
        )

        mlflow.log_param(
            "target_group_size",
            metadata[
                "target_group_size"
            ],
        )

        mlflow.log_param(
            "random_state",
            metadata["random_state"],
        )

        mlflow.log_metric(
            "number_of_groups",
            metadata["number_of_groups"],
        )

        mlflow.log_metric(
            "minimum_group_size",
            metadata[
                "minimum_group_size"
            ],
        )

        mlflow.log_metric(
            "mean_group_size",
            metadata[
                "mean_group_size"
            ],
        )

        mlflow.log_metric(
            "maximum_group_size",
            metadata[
                "maximum_group_size"
            ],
        )

        cluster_evaluation = metadata[
            "cluster_evaluation"
        ]

        for evaluation in (
            cluster_evaluation
        ):
            number_of_clusters = (
                evaluation[
                    "number_of_clusters"
                ]
            )

            mlflow.log_metric(
                (
                    "silhouette_k"
                    f"_{number_of_clusters}"
                ),
                evaluation["silhouette"],
            )

            mlflow.log_metric(
                (
                    "inertia_k"
                    f"_{number_of_clusters}"
                ),
                evaluation["inertia"],
            )

        log_existing_artifact(
            file_path=GROUPING_MODEL_PATH,
            artifact_path="models",
        )

        log_existing_artifact(
            file_path=GROUPING_METADATA_PATH,
            artifact_path="metadata",
        )

        log_existing_artifact(
            file_path=GROUP_ASSIGNMENTS_PATH,
            artifact_path="groups",
        )

        log_existing_artifact(
            file_path=GROUP_SUMMARY_PATH,
            artifact_path="groups",
        )


def configure_mlflow() -> None:
    """
    Configure MLflow avec une base SQLite
    locale compatible avec les versions
    récentes de MLflow.
    """
    database_path = (
        MLFLOW_DATABASE_PATH
        .resolve()
        .as_posix()
    )

    tracking_uri = (
        f"sqlite:///{database_path}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "MLflow tracking URI : "
        f"{tracking_uri}"
    )

def main() -> None:
    """
    Exécute le pipeline complet dans
    une expérience MLflow.
    """
    configure_mlflow()

    with mlflow.start_run(
        run_name="complete_pipeline",
    ) as parent_run:
        mlflow.log_param(
            "project",
            "adaptive-risk-management",
        )

        mlflow.log_param(
            "group_size",
            50,
        )

        mlflow.set_tag(
            "project_type",
            "continuous_learning",
        )

        mlflow.set_tag(
            "domain",
            "collaborative_insurance",
        )

        run_data_preparation()
        run_classification_training()
        run_cost_training()
        run_group_construction()

        print(
            "Pipeline MLflow terminé."
        )

        print(
            "Run parent : "
            f"{parent_run.info.run_id}"
        )


if __name__ == "__main__":
    main()
