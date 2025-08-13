# main.py

import logging
import os

from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import train_and_compare_models
from vis.visualizations import generate_visuals


def setup_logger(log_dir: str = "logs", log_file: str = "pipeline.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reruns
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        ch = logging.StreamHandler()

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def main():
    logger = setup_logger()
    logger.info("=== Pipeline start ===")

    raw_input = "insurance_claims.csv"
    raw_output = "data/extracted/insurance_claims_raw.csv"
    processed_output = "data/processed/insurance_claims_cleaned.csv"

    # 1) Extract
    try:
        df_extracted = extract_data(raw_input, raw_output)
        logger.info(f"Extracted → {raw_output} (rows={len(df_extracted)})")
    except Exception as e:
        logger.exception(f"Extract stage failed: {e}")
        df_extracted = None

    # 2) Transform
    try:
        if df_extracted is None:
            raise RuntimeError("No extracted DataFrame available.")
        df_transformed = transform_data(df_extracted)
        logger.info(f"Transformed dataframe shape: {df_transformed.shape}")
    except Exception as e:
        logger.exception(f"Transform stage failed: {e}")
        df_transformed = None

    # 3) Load
    try:
        if df_transformed is None:
            raise RuntimeError("No transformed DataFrame available.")
        load_data(df_transformed, processed_output)
        logger.info(f"Processed saved → {processed_output}")
    except Exception as e:
        logger.exception(f"Load stage failed: {e}")

    # 4) Model evaluation (Part 3 Type 1 testing)
    try:
        metrics_df = train_and_compare_models(processed_data_path=processed_output, logger=logger)
        logger.info("Model training & evaluation complete.")
    except Exception as e:
        logger.exception(f"Model stage failed: {e}")

    # 5) Visualizations (EDA/model visuals you already built)
    try:
        generate_visuals(processed_output)
        logger.info("Visualizations generated.")
    except Exception as e:
        logger.exception(f"Visualization stage failed: {e}")

    logger.info("=== Pipeline end ===")


if __name__ == "__main__":
    main()
