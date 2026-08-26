from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

NUMERICAL_FEATURES = [
    "Age_client",
    "age_of_car_M",
    "Car_power_M",
    "Insuredcapital_content_re",
    "Insuredcapital_continent_re",
    "Client_Seniority"
]

BINARY_FEATURES = [
    "gender",
    "Car_2ndDriver_M",
    "num_policiesC",
    "metro_code",
    "Policy_PaymentMethodA",
    "Policy_PaymentMethodH",
    "appartment"
]

FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + BINARY_FEATURES
)

CLASSIFICATION_TARGET = "Has_Claim"
FREQUENCY_TARGET = "Total_NClaims"
SEVERITY_TARGET = "Average_Claim_Severity"
COST_TARGET = "Total_Claims"

def load_processed_data(
    data_path: Path = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    data = pd.read_csv(
        data_path
    )

    return data

load_processed_data()

def validate_feature_columns(
    data: pd.DataFrame
) -> None:
    required_columns = (
        FEATURE_COLUMNS
        + [
            "PolID",
            "year",
            CLASSIFICATION_TARGET,
            FREQUENCY_TARGET,
            SEVERITY_TARGET,
            COST_TARGET
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes nécessaires absentes : "
            f"{missing_columns}"
        )

    if data[
        FEATURE_COLUMNS
    ].isnull().any().any():
        raise ValueError(
            "Les variables explicatives "
            "contiennent des valeurs manquantes."
        )

    if "Types" in FEATURE_COLUMNS:
        raise ValueError(
            "La variable Types ne doit pas "
            "être utilisée comme prédicteur."
        )

    print(
        "Validation des variables réussie."
    )

def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

def temporal_split(
    data: pd.DataFrame,
    target_column: str
) -> dict:
    train_mask = data["year"].isin(
        [1, 2, 3]
    )

    validation_mask = (
        data["year"] == 4
    )

    test_mask = (
        data["year"] == 5
    )

    split_data = {
        "X_train": data.loc[
            train_mask,
            FEATURE_COLUMNS
        ].copy(),

        "y_train": data.loc[
            train_mask,
            target_column
        ].copy(),

        "X_validation": data.loc[
            validation_mask,
            FEATURE_COLUMNS
        ].copy(),

        "y_validation": data.loc[
            validation_mask,
            target_column
        ].copy(),

        "X_test": data.loc[
            test_mask,
            FEATURE_COLUMNS
        ].copy(),

        "y_test": data.loc[
            test_mask,
            target_column
        ].copy()
    }

    return split_data


def validate_temporal_split(
    data: pd.DataFrame,
    split_data: dict
) -> None:
    total_split_rows = (
        len(split_data["X_train"])
        + len(split_data["X_validation"])
        + len(split_data["X_test"])
    )

    if total_split_rows != len(data):
        raise ValueError(
            "Certaines observations ne sont "
            "pas affectées au découpage."
        )

    if len(split_data["X_train"]) == 0:
        raise ValueError(
            "L'entraînement est vide."
        )

    if len(split_data["X_validation"]) == 0:
        raise ValueError(
            "La validation est vide."
        )

    if len(split_data["X_test"]) == 0:
        raise ValueError(
            "Le test est vide."
        )

    print(
        "Validation du découpage temporel réussie."
    )

def print_split_summary(
    split_data: dict
) -> None:
    summary = pd.DataFrame({
        "Ensemble": [
            "Entraînement",
            "Validation",
            "Test"
        ],
        "Nombre_observations": [
            len(split_data["X_train"]),
            len(split_data["X_validation"]),
            len(split_data["X_test"])
        ],
        "Taux_cible_moyen": [
            split_data["y_train"].mean(),
            split_data["y_validation"].mean(),
            split_data["y_test"].mean()
        ]
    })

    print(
        summary.to_string(
            index=False
        )
    )

def main() -> None:
    data = load_processed_data()

    validate_feature_columns(
        data
    )

    split_data = temporal_split(
        data=data,
        target_column=CLASSIFICATION_TARGET
    )

    validate_temporal_split(
        data=data,
        split_data=split_data
    )

    print_split_summary(
        split_data
    )

if __name__ == "__main__":
    main()
