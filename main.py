from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import train_model

def main():
    print("Starting pipeline...")

    raw_input = "insurance_claims.csv"
    raw_output = "data/extracted/insurance_claims_raw.csv"
    processed_output = "data/processed/insurance_claims_cleaned.csv"

    # ETL
    df = extract_data(raw_input, raw_output)
    df_clean = transform_data(df)
    load_data(df_clean, processed_output)

    # Model training
    model = train_model(processed_output)

if __name__ == "__main__":
    main()
