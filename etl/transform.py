import logging
import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def transform_data(df: pd.DataFrame,
                   logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Clean and prepare the dataframe for modeling.

    Notes:
    - Leaves feature encoding/scaling to the modeling pipeline (ColumnTransformer).
    - Ensures target 'fraud_reported' is 0/1 integers.
    - Drops exact duplicates and rows with null target.
    """
    if logger: logger.info("[Transform] Starting")
    try:
        tdf = _standardize_columns(df)

        # Normalize target to 0/1 (supports original 'Y'/'N' or 0/1)
        if "fraud_reported" in tdf.columns:
            if tdf["fraud_reported"].dtype == object:
                tdf["fraud_reported"] = (
                    tdf["fraud_reported"].str.strip().str.upper().map({"Y": 1, "N": 0})
                )
            # Coerce to Int64 (nullable ints), then drop nulls
            tdf["fraud_reported"] = pd.to_numeric(tdf["fraud_reported"], errors="coerce").astype("Int64")
            tdf = tdf.dropna(subset=["fraud_reported"])
            tdf["fraud_reported"] = tdf["fraud_reported"].astype(int)
        else:
            raise ValueError("Target column 'fraud_reported' not found after standardization.")

        # Basic de-duplication
        before = len(tdf)
        tdf = tdf.drop_duplicates()
        after = len(tdf)
        if logger and after != before:
            logger.info(f"[Transform] Dropped duplicates: {before - after}")


        if logger: logger.info(f"[Transform] Done. Shape: {tdf.shape}")
        return tdf

    except Exception as e:
        if logger: logger.exception(f"[Transform] Failed: {e}")
        raise
