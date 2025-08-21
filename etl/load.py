import os
import logging
import pandas as pd


def load_data(df: pd.DataFrame,
              processed_out_csv: str,
              logger: logging.Logger | None = None) -> None:
    """Persist the processed dataset to CSV."""
    if logger: logger.info("[Load] Starting")
    try:
        os.makedirs(os.path.dirname(processed_out_csv), exist_ok=True)
        df.to_csv(processed_out_csv, index=False)
        if logger: logger.info(f"[Load] Saved processed → {processed_out_csv}")
    except Exception as e:
        if logger: logger.exception(f"[Load] Failed: {e}")
        raise
