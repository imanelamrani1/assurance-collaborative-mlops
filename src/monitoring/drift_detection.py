import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.features.build_features import (
    BINARY_FEATURES,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    load_processed_data,
)


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

DRIFT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "drift_report.csv"
)

DRIFT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "drift_summary.json"
)

REFERENCE_YEARS = [
    1,
    2,
    3,
    4,
]

CURRENT_YEAR = 5

PSI_STABLE_LIMIT = 0.10
PSI_SIGNIFICANT_LIMIT = 0.25

PR_AUC_DROP_LIMIT = 0.05
TARGET_RATE_RELATIVE_CHANGE_LIMIT = 0.20


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


def classify_psi(
    psi_value: float,
) -> str:
    """
    Classe le niveau du PSI.
    """
    if psi_value < PSI_STABLE_LIMIT:
        return "stable"

    if psi_value < PSI_SIGNIFICANT_LIMIT:
        return "moderate"

    return "significant"


def calculate_numerical_psi(
    reference_values: pd.Series,
    current_values: pd.Series,
    number_of_bins: int = 10,
) -> float:
    """
    Calcule le Population Stability Index
    d'une variable numérique.
    """
    reference_array = (
        reference_values
        .dropna()
        .to_numpy()
    )

    current_array = (
        current_values
        .dropna()
        .to_numpy()
    )

    if len(reference_array) == 0:
        raise ValueError(
            "La distribution de référence "
            "est vide."
        )

    if len(current_array) == 0:
        raise ValueError(
            "La distribution actuelle "
            "est vide."
        )

    quantiles = np.linspace(
        0,
        1,
        number_of_bins + 1,
    )

    bin_edges = np.quantile(
        reference_array,
        quantiles,
    )

    bin_edges = np.unique(
        bin_edges
    )

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_array,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current_array,
        bins=bin_edges,
    )

    reference_proportions = (
        reference_counts
        / reference_counts.sum()
    )

    current_proportions = (
        current_counts
        / current_counts.sum()
    )

    epsilon = 1e-6

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    psi_value = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi_value)


def calculate_categorical_psi(
    reference_values: pd.Series,
    current_values: pd.Series,
) -> float:
    """
    Calcule le PSI d'une variable
    catégorielle ou binaire.
    """
    categories = sorted(
        set(
            reference_values
            .dropna()
            .unique()
        )
        | set(
            current_values
            .dropna()
            .unique()
        )
    )

    reference_proportions = (
        reference_values
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    current_proportions = (
        current_values
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0,
        )
        .to_numpy()
    )

    epsilon = 1e-6

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    psi_value = np.sum(
        (
            current_proportions
            - reference_proportions
        )
        * np.log(
            current_proportions
            / reference_proportions
        )
    )

    return float(psi_value)


def calculate_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule le PSI de toutes les variables
    utilisées par les modèles.
    """
    drift_results = []

    for feature in NUMERICAL_FEATURES:
        psi_value = calculate_numerical_psi(
            reference_values=(
                reference_data[feature]
            ),
            current_values=(
                current_data[feature]
            ),
        )

        drift_results.append({
            "Feature": feature,
            "Feature_Type": "numerical",
            "PSI": psi_value,
            "Drift_Level": (
                classify_psi(
                    psi_value
                )
            ),
        })

    for feature in BINARY_FEATURES:
        psi_value = (
            calculate_categorical_psi(
                reference_values=(
                    reference_data[feature]
                ),
                current_values=(
                    current_data[feature]
                ),
            )
        )

        drift_results.append({
            "Feature": feature,
            "Feature_Type": "binary",
            "PSI": psi_value,
            "Drift_Level": (
                classify_psi(
                    psi_value
                )
            ),
        })

    drift_report = pd.DataFrame(
        drift_results
    )

    drift_report = (
        drift_report
        .sort_values(
            by="PSI",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return drift_report


def evaluate_current_performance(
    current_data: pd.DataFrame,
    model,
) -> dict:
    """
    Évalue les probabilités du modèle
    sur l'année actuelle.
    """
    X_current = current_data[
        FEATURE_COLUMNS
    ].copy()

    y_current = current_data[
        CLASSIFICATION_TARGET
    ].copy()

    probabilities = (
        model.predict_proba(
            X_current
        )[:, 1]
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_current,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_current,
                probabilities,
            )
        ),
        "target_rate": float(
            y_current.mean()
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
    }


def calculate_relative_change(
    reference_value: float,
    current_value: float,
) -> float:
    """
    Calcule la variation relative.
    """
    if reference_value == 0:
        return 0.0

    return float(
        (
            current_value
            - reference_value
        )
        / reference_value
    )


def build_drift_summary(
    data: pd.DataFrame,
    drift_report: pd.DataFrame,
    current_performance: dict,
    metadata: dict,
) -> dict:
    """
    Produit la décision de réentraînement.
    """
    selected_model_name = metadata[
        "model_name"
    ]

    reference_validation_metrics = (
        metadata[
            "validation_results"
        ][selected_model_name]
    )

    reference_pr_auc = float(
        reference_validation_metrics[
            "pr_auc"
        ]
    )

    current_pr_auc = float(
        current_performance[
            "pr_auc"
        ]
    )

    pr_auc_drop = (
        reference_pr_auc
        - current_pr_auc
    )

    reference_mask = data[
        "year"
    ].isin(REFERENCE_YEARS)

    reference_target_rate = float(
        data.loc[
            reference_mask,
            CLASSIFICATION_TARGET,
        ].mean()
    )

    current_target_rate = float(
        current_performance[
            "target_rate"
        ]
    )

    target_rate_relative_change = (
        calculate_relative_change(
            reference_value=(
                reference_target_rate
            ),
            current_value=(
                current_target_rate
            ),
        )
    )

    significant_features = (
        drift_report.loc[
            drift_report[
                "Drift_Level"
            ] == "significant",
            "Feature",
        ].tolist()
    )

    moderate_features = (
        drift_report.loc[
            drift_report[
                "Drift_Level"
            ] == "moderate",
            "Feature",
        ].tolist()
    )

    feature_drift_trigger = (
        len(significant_features) > 0
    )

    performance_trigger = (
        pr_auc_drop
        > PR_AUC_DROP_LIMIT
    )

    target_drift_trigger = (
        abs(
            target_rate_relative_change
        )
        > TARGET_RATE_RELATIVE_CHANGE_LIMIT
    )

    retraining_required = any([
        feature_drift_trigger,
        performance_trigger,
        target_drift_trigger,
    ])

    return {
        "reference_years": REFERENCE_YEARS,
        "current_year": CURRENT_YEAR,
        "reference_pr_auc": (
            reference_pr_auc
        ),
        "current_pr_auc": (
            current_pr_auc
        ),
        "pr_auc_drop": float(
            pr_auc_drop
        ),
        "reference_target_rate": (
            reference_target_rate
        ),
        "current_target_rate": (
            current_target_rate
        ),
        "target_rate_relative_change": (
            target_rate_relative_change
        ),
        "significant_drift_features": (
            significant_features
        ),
        "moderate_drift_features": (
            moderate_features
        ),
        "feature_drift_trigger": (
            feature_drift_trigger
        ),
        "performance_trigger": (
            performance_trigger
        ),
        "target_drift_trigger": (
            target_drift_trigger
        ),
        "retraining_required": (
            retraining_required
        ),
        "thresholds": {
            "psi_stable_limit": (
                PSI_STABLE_LIMIT
            ),
            "psi_significant_limit": (
                PSI_SIGNIFICANT_LIMIT
            ),
            "pr_auc_drop_limit": (
                PR_AUC_DROP_LIMIT
            ),
            "target_rate_relative_change_limit": (
                TARGET_RATE_RELATIVE_CHANGE_LIMIT
            ),
        },
    }


def save_reports(
    drift_report: pd.DataFrame,
    drift_summary: dict,
) -> None:
    """
    Enregistre les rapports de drift.
    """
    DRIFT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_report.to_csv(
        DRIFT_REPORT_PATH,
        index=False,
    )

    with open(
        DRIFT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            drift_summary,
            summary_file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Rapport PSI : "
        f"{DRIFT_REPORT_PATH}"
    )

    print(
        f"Résumé du drift : "
        f"{DRIFT_SUMMARY_PATH}"
    )


def main() -> None:
    """
    Exécute la surveillance du drift.
    """
    if not CLASSIFICATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Le modèle de classification "
            "est absent."
        )

    data = load_processed_data()

    model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    metadata = read_json(
        CLASSIFICATION_METADATA_PATH
    )

    reference_data = data.loc[
        data["year"].isin(
            REFERENCE_YEARS
        )
    ].copy()

    current_data = data.loc[
        data["year"] == CURRENT_YEAR
    ].copy()

    if reference_data.empty:
        raise ValueError(
            "Les données de référence "
            "sont vides."
        )

    if current_data.empty:
        raise ValueError(
            "Les données actuelles "
            "sont vides."
        )

    drift_report = (
        calculate_feature_drift(
            reference_data=reference_data,
            current_data=current_data,
        )
    )

    current_performance = (
        evaluate_current_performance(
            current_data=current_data,
            model=model,
        )
    )

    drift_summary = (
        build_drift_summary(
            data=data,
            drift_report=drift_report,
            current_performance=(
                current_performance
            ),
            metadata=metadata,
        )
    )

    print(
        "\nVariables avec le plus de drift :"
    )

    print(
        drift_report.head(10).to_string(
            index=False
        )
    )

    print(
        "\nRésumé de la surveillance :"
    )

    print(
        json.dumps(
            drift_summary,
            indent=4,
            ensure_ascii=False,
        )
    )

    if drift_summary[
        "retraining_required"
    ]:
        print(
            "\nALERTE : un réentraînement "
            "est recommandé."
        )
    else:
        print(
            "\nAucun réentraînement requis."
        )

    save_reports(
        drift_report=drift_report,
        drift_summary=drift_summary,
    )


if __name__ == "__main__":
    main()
