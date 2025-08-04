import os
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import train_and_evaluate_model
from vis.visualizations import generate_visuals

def main():
    print("Starting pipeline...")

    # Paths
    raw_input = "insurance_claims.csv"  # CSV must exist in project root or correct path
    raw_output = "data/extracted/insurance_claims_raw.csv"
    processed_output = "data/processed/insurance_claims_cleaned.csv"

    # Step 1: Extract
    df_extracted = extract_data(raw_input, raw_output)

    # Step 2: Transform
    df_transformed = transform_data(df_extracted)
    
    # Step 3: Load
    load_data(df_transformed, processed_output)

    # Step 4: Model
    train_and_evaluate_model(processed_output)

    # Step 5: Visualizations
    generate_visuals(processed_output)

if __name__ == "__main__":
    main()
