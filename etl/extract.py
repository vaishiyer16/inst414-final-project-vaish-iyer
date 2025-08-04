import pandas as pd
import os

def extract_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Extracted data saved to {output_path}")
    return df
