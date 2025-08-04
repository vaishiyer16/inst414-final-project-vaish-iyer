import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace '?' with NaN
    df.replace("?", pd.NA, inplace=True)

    # Drop empty column
    if "_c39" in df.columns:
        df.drop(columns=["_c39"], inplace=True)

    # Encode fraud_reported (Y/N → 1/0)
    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # Drop policy_number, incident_location (identifiers)
    df.drop(columns=["policy_number", "incident_location"], inplace=True)

    print(f"[INFO] Data transformed. Shape: {df.shape}")
    return df
