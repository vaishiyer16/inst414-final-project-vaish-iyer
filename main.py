import os
import logging

from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import train_and_compare_models
from vis.visualizations import generate_visuals


def setup_logger(log_dir: str = "logs", log_file: str = "pipeline.log") -> logging.Logger:
    """Configure a rotating file + console logger (simple)."""
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

    # Prefer a URL (true Extract). Use your real source if available.
    # Example placeholder (replace with your dataset URL if you have one):
    # raw_input = "https://YOUR-SOURCE/insurance_claims.csv"
    # If no URL is available, the extractor will still work with a local path:
    raw_input = "insurance_claims.csv"

    raw_output = "data/extracted/insurance_claims_raw.csv"
    processed_output = "data/processed/insurance_claims_cleaned.csv"

    # 1) Extract
    try:
        df_extracted = extract_data(raw_input, raw_output, logger=logger)
    except Exception:
        df_extracted = None

    # 2) Transform
    try:
        if df_extracted is None:
            raise RuntimeError("No extracted DataFrame available.")
        df_transformed = transform_data(df_extracted, logger=logger)
    except Exception:
        df_transformed = None

    # 3) Load
    try:
        if df_transformed is None:
            raise RuntimeError("No transformed DataFrame available.")
        load_data(df_transformed, processed_output, logger=logger)
    except Exception:
        pass

    # 4) Model evaluation (Type 1 testing)
    try:
        train_and_compare_models(processed_output, logger=logger)
        logger.info("Model training & evaluation complete.")
    except Exception:
        logger.exception("Model stage failed.")

    # 5) Visualizations
    try:
        generate_visuals(processed_output, logger=logger)
    except Exception:
        logger.exception("Visualization stage failed.")

    logger.info("=== Pipeline end ===")


if __name__ == "__main__":
    main()
