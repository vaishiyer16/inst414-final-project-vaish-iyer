import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_evaluate_model(processed_data_path: str):
    """
    Loads processed data, trains a Random Forest model, and prints evaluation results.
    """
    # Load processed data
    df = pd.read_csv(processed_data_path)

    # Debug: Show unique values in fraud_reported
    print("[DEBUG] Unique values in 'fraud_reported':", df["fraud_reported"].unique())

    # Drop rows where the target is missing
    df = df.dropna(subset=["fraud_reported"])

    # Drop 'policy_number' if it exists
    if "policy_number" in df.columns:
        df = df.drop(columns=["policy_number"])

    # Select numeric features and target
    X = df.select_dtypes(include=["number"]).drop(columns=["fraud_reported"])
    y = df["fraud_reported"]

    # Debug: Show shape
    print(f"[DEBUG] Final dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))
