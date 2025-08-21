import os
import logging
import pandas as pd
from urllib.parse import urlparse
from urllib.request import urlretrieve


def _is_url(path: str) -> bool:
    """Return True if path looks like an HTTP(S) URL."""
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def extract_data(input_path_or_url: str,
                 raw_out_csv: str,
                 logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Extract stage:
      - If given an HTTP(S) URL, download and read.
      - Else, read local CSV.
    Saves the raw CSV to raw_out_csv and returns a DataFrame.
    """
    if logger: logger.info(f"[Extract] Source: {input_path_or_url}")
    os.makedirs(os.path.dirname(raw_out_csv), exist_ok=True)

    try:
        if _is_url(input_path_or_url):
            if logger: logger.info("[Extract] Detected URL. Downloading file...")
            tmp_path = raw_out_csv + ".download.tmp.csv"
            urlretrieve(input_path_or_url, tmp_path)
            df = pd.read_csv(tmp_path)
            df.to_csv(raw_out_csv, index=False)
            if logger: logger.info(f"[Extract] Downloaded → {raw_out_csv} (rows={len(df)})")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Local CSV fallback
            df = pd.read_csv(input_path_or_url)
            df.to_csv(raw_out_csv, index=False)
            if logger: logger.info(f"[Extract] Loaded local CSV → {raw_out_csv} (rows={len(df)})")

        return df

    except Exception as e:
        if logger: logger.exception(f"[Extract] Failed: {e}")
        raise
