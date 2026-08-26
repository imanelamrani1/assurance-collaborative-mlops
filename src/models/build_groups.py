import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features.build_features import (
    FEATURE_COLUMNS,
    load_processed_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_SCORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_scores_year5.csv"
)

CLUSTER_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_assignments_year5.csv"
)

GROUP_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "collaborative_group_assignments.csv"
)

GROUP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "collaborative_group_summary.csv"
)

GROUPING_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "grouping_bundle.joblib"
)

GROUPING_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "grouping_metadata.json"
)

TARGET_GROUP_SIZE = 50
RANDOM_STATE = 42


def load_risk_scores() -> pd.DataFrame:
    """
    Charge les scores de risque de l'année 5.
    """
    if not RISK_SCORE_PATH.exists():
        raise FileNotFoundError(
            "Le fichier risk_scores_year5.csv "
            "est absent. Exécutez d'abord "
            "src.models.train_cost."
        )

    risk_scores = pd.read_csv(
        RISK_SCORE_PATH
    )

    return risk_scores


def prepare_clustering_data(
    data: pd.DataFrame,
    risk_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sélectionne les profils de l'année 5
    et ajoute les scores prédits.
    """
    year5_profiles = data.loc[
        data["year"] == 5
    ].copy()

    if not year5_profiles[
        "PolID"
    ].is_unique:
        raise ValueError(
            "PolID n'est pas unique "
            "pour l'année 5."
        )

    if not risk_scores[
        "PolID"
    ].is_unique:
        raise ValueError(
            "PolID n'est pas unique dans "
            "le fichier des scores."
        )

    predicted_columns = [
        "PolID",
        "year",
        "Predicted_Claim_Probability",
        "Predicted_Frequency",
        "Predicted_Severity",
        "Predicted_Expected_Cost",
        "Risk_Decile",
        "Risk_Level",
    ]

    missing_score_columns = [
        column
        for column in predicted_columns
        if column not in risk_scores.columns
    ]

    if missing_score_columns:
        raise ValueError(
            "Colonnes de score absentes : "
            f"{missing_score_columns}"
        )

    clustering_data = (
        year5_profiles.merge(
            risk_scores[
                predicted_columns
            ],
            on=[
                "PolID",
                "year",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if len(clustering_data) != len(
        year5_profiles
    ):
        raise ValueError(
            "Certains profils de l'année 5 "
            "n'ont pas de score de risque."
        )

    if clustering_data[
        FEATURE_COLUMNS
    ].isnull().any().any():
        raise ValueError(
            "Les variables de clustering "
            "contiennent des valeurs manquantes."
        )

    return clustering_data


def select_number_of_clusters(
    scaled_features: np.ndarray,
) -> tuple[
    int,
    list,
]:
    """
    Teste K entre 2 et 10 et sélectionne
    le meilleur score de silhouette.
    """
    evaluation_results = []

    sample_size = min(
        5000,
        len(scaled_features),
    )

    for number_of_clusters in range(
        2,
        11,
    ):
        print(
            "Évaluation de K = "
            f"{number_of_clusters}"
        )

        candidate_model = KMeans(
            n_clusters=(
                number_of_clusters
            ),
            n_init=20,
            random_state=RANDOM_STATE,
        )

        candidate_labels = (
            candidate_model.fit_predict(
                scaled_features
            )
        )

        silhouette = silhouette_score(
            scaled_features,
            candidate_labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )

        evaluation_results.append({
            "number_of_clusters": int(
                number_of_clusters
            ),
            "inertia": float(
                candidate_model.inertia_
            ),
            "silhouette": float(
                silhouette
            ),
        })

    best_result = max(
        evaluation_results,
        key=lambda result: (
            result["silhouette"]
        ),
    )

    best_number_of_clusters = int(
        best_result[
            "number_of_clusters"
        ]
    )

    return (
        best_number_of_clusters,
        evaluation_results,
    )


def fit_clustering_model(
    clustering_data: pd.DataFrame,
) -> tuple:
    """
    Standardise les variables, sélectionne K
    et entraîne le K-Means final.
    """
    feature_data = clustering_data[
        FEATURE_COLUMNS
    ].copy()

    scaler = StandardScaler()

    scaled_features = (
        scaler.fit_transform(
            feature_data
        )
    )

    (
        best_number_of_clusters,
        evaluation_results,
    ) = select_number_of_clusters(
        scaled_features
    )

    print(
        "Nombre de clusters sélectionné : "
        f"{best_number_of_clusters}"
    )

    kmeans_model = KMeans(
        n_clusters=(
            best_number_of_clusters
        ),
        n_init=50,
        random_state=RANDOM_STATE,
    )

    cluster_labels = (
        kmeans_model.fit_predict(
            scaled_features
        )
        + 1
    )

    result_data = (
        clustering_data.copy()
    )

    result_data[
        "Cluster_Label"
    ] = cluster_labels

    return (
        result_data,
        scaler,
        kmeans_model,
        evaluation_results,
        best_number_of_clusters,
    )


def assign_balanced_groups(
    cluster_data: pd.DataFrame,
    target_size: int,
) -> pd.DataFrame:
    """
    Répartit les membres d'un cluster dans
    des groupes de taille similaire tout en
    équilibrant le coût attendu total.
    """
    local_data = (
        cluster_data
        .sort_values(
            by="Predicted_Expected_Cost",
            ascending=False,
        )
        .copy()
    )

    number_of_members = len(
        local_data
    )

    number_of_groups = max(
        1,
        int(
            np.floor(
                number_of_members
                / target_size
                + 0.5
            )
        ),
    )

    base_capacity = (
        number_of_members
        // number_of_groups
    )

    remaining_members = (
        number_of_members
        % number_of_groups
    )

    capacities = np.full(
        number_of_groups,
        base_capacity,
        dtype=int,
    )

    capacities[
        :remaining_members
    ] += 1

    group_costs = np.zeros(
        number_of_groups,
        dtype=float,
    )

    group_counts = np.zeros(
        number_of_groups,
        dtype=int,
    )

    assignments = []

    for expected_cost in local_data[
        "Predicted_Expected_Cost"
    ].to_numpy():
        eligible_groups = np.where(
            group_counts < capacities
        )[0]

        if len(eligible_groups) == 0:
            raise RuntimeError(
                "Aucun groupe éligible "
                "pour cette observation."
            )

        eligible_costs = group_costs[
            eligible_groups
        ]

        selected_group = int(
            eligible_groups[
                np.argmin(
                    eligible_costs
                )
            ]
        )

        assignments.append(
            selected_group + 1
        )

        group_costs[
            selected_group
        ] += float(expected_cost)

        group_counts[
            selected_group
        ] += 1

    local_data[
        "Local_Group"
    ] = assignments

    cluster_label = int(
        local_data[
            "Cluster_Label"
        ].iloc[0]
    )

    local_data[
        "Collaborative_Group"
    ] = local_data[
        "Local_Group"
    ].apply(
        lambda group_number: (
            f"C{cluster_label:02d}"
            f"_G{int(group_number):03d}"
        )
    )

    return local_data


def build_collaborative_groups(
    clustered_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique l'affectation équilibrée
    à chaque cluster.
    """
    group_parts = []

    cluster_labels = sorted(
        clustered_data[
            "Cluster_Label"
        ].unique()
    )

    for cluster_label in cluster_labels:
        print(
            "Création des groupes du cluster "
            f"{cluster_label}"
        )

        cluster_subset = clustered_data[
            clustered_data[
                "Cluster_Label"
            ] == cluster_label
        ].copy()

        assigned_subset = (
            assign_balanced_groups(
                cluster_data=cluster_subset,
                target_size=(
                    TARGET_GROUP_SIZE
                ),
            )
        )

        group_parts.append(
            assigned_subset
        )

    collaborative_groups = pd.concat(
        group_parts,
        ignore_index=True,
    )

    if len(collaborative_groups) != len(
        clustered_data
    ):
        raise ValueError(
            "Certaines observations n'ont "
            "pas été affectées."
        )

    if collaborative_groups[
        "Collaborative_Group"
    ].isnull().any():
        raise ValueError(
            "Certaines affectations "
            "de groupe sont absentes."
        )

    return collaborative_groups


def create_group_summary(
    collaborative_groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule les caractéristiques de chaque
    groupe collaboratif.
    """
    summary = (
        collaborative_groups
        .groupby(
            [
                "Cluster_Label",
                "Collaborative_Group",
            ]
        )
        .agg(
            Number_of_Members=(
                "PolID",
                "nunique",
            ),
            Mean_Claim_Probability=(
                "Predicted_Claim_Probability",
                "mean",
            ),
            Mean_Predicted_Frequency=(
                "Predicted_Frequency",
                "mean",
            ),
            Mean_Predicted_Severity=(
                "Predicted_Severity",
                "mean",
            ),
            Mean_Expected_Cost=(
                "Predicted_Expected_Cost",
                "mean",
            ),
            Total_Expected_Cost=(
                "Predicted_Expected_Cost",
                "sum",
            ),
            Real_Claim_Count=(
                "Total_NClaims",
                "sum",
            ),
            Real_Total_Cost=(
                "Total_Claims",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary


def validate_groups(
    collaborative_groups: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    """
    Valide les principales contraintes.
    """
    clusters_per_group = (
        collaborative_groups
        .groupby(
            "Collaborative_Group"
        )["Cluster_Label"]
        .nunique()
    )

    if not (
        clusters_per_group == 1
    ).all():
        raise ValueError(
            "Un groupe contient des membres "
            "de plusieurs clusters."
        )

    assigned_members = int(
        group_summary[
            "Number_of_Members"
        ].sum()
    )

    expected_members = int(
        collaborative_groups[
            "PolID"
        ].nunique()
    )

    if assigned_members != expected_members:
        raise ValueError(
            "Le nombre de membres du résumé "
            "est incohérent."
        )

    print(
        "Validation des groupes réussie."
    )

    print(
        "Nombre de groupes : "
        f"{len(group_summary)}"
    )

    print(
        "Taille minimale : "
        f"{group_summary['Number_of_Members'].min()}"
    )

    print(
        "Taille moyenne : "
        f"{group_summary['Number_of_Members'].mean():.2f}"
    )

    print(
        "Taille maximale : "
        f"{group_summary['Number_of_Members'].max()}"
    )


def save_outputs(
    collaborative_groups: pd.DataFrame,
    group_summary: pd.DataFrame,
    scaler: StandardScaler,
    kmeans_model: KMeans,
    evaluation_results: list,
    best_number_of_clusters: int,
) -> None:
    """
    Enregistre les affectations, résumés,
    modèles et métadonnées.
    """
    output_columns = [
        "PolID",
        "year",
        "Cluster_Label",
        "Collaborative_Group",
        "Predicted_Claim_Probability",
        "Predicted_Frequency",
        "Predicted_Severity",
        "Predicted_Expected_Cost",
        "Risk_Decile",
        "Risk_Level",
        "Has_Claim",
        "Total_NClaims",
        "Total_Claims",
    ]

    cluster_output_columns = [
        "PolID",
        "year",
        "Cluster_Label",
        "Predicted_Claim_Probability",
        "Predicted_Expected_Cost",
        "Risk_Decile",
        "Risk_Level",
    ]

    GROUP_ASSIGNMENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUPING_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collaborative_groups[
        cluster_output_columns
    ].to_csv(
        CLUSTER_ASSIGNMENTS_PATH,
        index=False,
    )

    collaborative_groups[
        output_columns
    ].sort_values(
        by=[
            "Cluster_Label",
            "Collaborative_Group",
            "Predicted_Expected_Cost",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).to_csv(
        GROUP_ASSIGNMENTS_PATH,
        index=False,
    )

    group_summary.to_csv(
        GROUP_SUMMARY_PATH,
        index=False,
    )

    grouping_bundle = {
        "features": FEATURE_COLUMNS,
        "scaler": scaler,
        "kmeans_model": kmeans_model,
        "target_group_size": (
            TARGET_GROUP_SIZE
        ),
    }

    joblib.dump(
        grouping_bundle,
        GROUPING_MODEL_PATH,
    )

    metadata = {
        "year": 5,
        "features": FEATURE_COLUMNS,
        "best_number_of_clusters": (
            best_number_of_clusters
        ),
        "target_group_size": (
            TARGET_GROUP_SIZE
        ),
        "random_state": RANDOM_STATE,
        "cluster_evaluation": (
            evaluation_results
        ),
        "number_of_groups": int(
            len(group_summary)
        ),
        "minimum_group_size": int(
            group_summary[
                "Number_of_Members"
            ].min()
        ),
        "mean_group_size": float(
            group_summary[
                "Number_of_Members"
            ].mean()
        ),
        "maximum_group_size": int(
            group_summary[
                "Number_of_Members"
            ].max()
        ),
    }

    with open(
        GROUPING_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "Affectations enregistrées : "
        f"{GROUP_ASSIGNMENTS_PATH}"
    )

    print(
        "Résumé enregistré : "
        f"{GROUP_SUMMARY_PATH}"
    )

    print(
        "Bundle de clustering enregistré : "
        f"{GROUPING_MODEL_PATH}"
    )


def main() -> None:
    """
    Exécute tout le pipeline de création
    des groupes collaboratifs.
    """
    print(
        "Chargement des données..."
    )

    data = load_processed_data()
    risk_scores = load_risk_scores()

    clustering_data = (
        prepare_clustering_data(
            data=data,
            risk_scores=risk_scores,
        )
    )

    print(
        "Nombre de profils à segmenter : "
        f"{len(clustering_data)}"
    )

    (
        clustered_data,
        scaler,
        kmeans_model,
        evaluation_results,
        best_number_of_clusters,
    ) = fit_clustering_model(
        clustering_data
    )

    collaborative_groups = (
        build_collaborative_groups(
            clustered_data
        )
    )

    group_summary = (
        create_group_summary(
            collaborative_groups
        )
    )

    validate_groups(
        collaborative_groups=(
            collaborative_groups
        ),
        group_summary=group_summary,
    )

    save_outputs(
        collaborative_groups=(
            collaborative_groups
        ),
        group_summary=group_summary,
        scaler=scaler,
        kmeans_model=kmeans_model,
        evaluation_results=(
            evaluation_results
        ),
        best_number_of_clusters=(
            best_number_of_clusters
        ),
    )


if __name__ == "__main__":
    main()
