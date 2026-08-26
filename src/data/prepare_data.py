from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "data_ex.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

Path(__file__).resolve().parents[2]

def load_raw_data(
    data_path: Path
) -> pd.DataFrame:
    data = pd.read_csv(
        data_path
    )

    print(
        f"Données chargées : "
        f"{data.shape[0]} lignes, "
        f"{data.shape[1]} colonnes."
    )

    return data

def validate_raw_data(
    data: pd.DataFrame
) -> None:
    required_columns = [
        "PolID",
        "year",
        "gender",
        "Age_client",
        "age_of_car_M",
        "Car_power_M",
        "NClaims1",
        "NClaims2",
        "Claims1",
        "Claims2"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{missing_columns}"
        )
        if data.empty:
           raise ValueError(
            "Le dataset brut est vide."
        )

    if data.duplicated(
        subset=["PolID", "year"]
    ).any():
        raise ValueError(
            "La combinaison PolID-year "
            "contient des doublons."
        )

    non_negative_columns = [
        "Age_client",
        "age_of_car_M",
        "Car_power_M",
        "NClaims1",
        "NClaims2",
        "Claims1",
        "Claims2"
    ]

    for column in non_negative_columns:
        if (data[column] < 0).any():
            raise ValueError(
                f"Valeur négative dans {column}."
            )

    print(
        "Validation des données brutes réussie."
    )

def clean_structure(
    data: pd.DataFrame
) -> pd.DataFrame:
    cleaned_data = data.copy()

    cleaned_data = cleaned_data.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    cleaned_data = (
        cleaned_data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return cleaned_data

def create_targets(
    data: pd.DataFrame
) -> pd.DataFrame:
    prepared_data = data.copy()

    prepared_data[
        "Total_NClaims"
    ] = (
        prepared_data["NClaims1"]
        + prepared_data["NClaims2"]
    )

    prepared_data[
        "Total_Claims"
    ] = (
        prepared_data["Claims1"]
        + prepared_data["Claims2"]
    )

    prepared_data[
        "Has_Claim"
    ] = (
        prepared_data[
            "Total_NClaims"
        ] > 0
    ).astype("int64")

    prepared_data[
        "Has_Paid_Claim"
    ] = (
        prepared_data[
            "Total_Claims"
        ] > 0
    ).astype("int64")

    average_severity = np.full(
        len(prepared_data),
        np.nan,
        dtype=float
    )

    np.divide(
        prepared_data[
            "Total_Claims"
        ].to_numpy(),
        prepared_data[
            "Total_NClaims"
        ].to_numpy(),
        out=average_severity,
        where=(
            prepared_data[
                "Total_NClaims"
            ].to_numpy() > 0
        )
    )

    prepared_data[
        "Average_Claim_Severity"
    ] = average_severity

    return prepared_data

def validate_prepared_data(
    data: pd.DataFrame
) -> None:
    if "Unnamed: 0" in data.columns:
        raise ValueError(
            "La colonne Unnamed: 0 "
            "n'a pas été supprimée."
        )

    if data.duplicated(
        subset=["PolID", "year"]
    ).any():
        raise ValueError(
            "Doublons PolID-year détectés."
        )

    if not data[
        "Has_Claim"
    ].isin([0, 1]).all():
        raise ValueError(
            "Has_Claim doit contenir "
            "uniquement 0 et 1."
        )

    if (data["Total_NClaims"] < 0).any():
        raise ValueError(
            "Total_NClaims contient "
            "des valeurs négatives."
        )

    if (data["Total_Claims"] < 0).any():
        raise ValueError(
            "Total_Claims contient "
            "des valeurs négatives."
        )

    print(
        "Validation des données préparées réussie."
    )

def save_prepared_data(
    data: pd.DataFrame,
    output_path: Path
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_path,
        index=False
    )

    print(
        f"Données enregistrées dans : "
        f"{output_path}"
    )

def main() -> None:
    raw_data = load_raw_data(
        RAW_DATA_PATH
    )

    validate_raw_data(
        raw_data
    )

    cleaned_data = clean_structure(
        raw_data
    )

    prepared_data = create_targets(
        cleaned_data
    )

    validate_prepared_data(
        prepared_data
    )

    save_prepared_data(
        prepared_data,
        PROCESSED_DATA_PATH
    )

if __name__ == "__main__":
    main()
