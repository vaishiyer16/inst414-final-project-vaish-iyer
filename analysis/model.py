import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(processed_data_path: str):
    """
    Trains a Random Forest model on the cleaned dataset and prints classification metrics.

    Args:
        processed_data_path (str): Path to cleaned CSV
    """
    # Load processed data
    df = pd.read_csv(processed_data_path)

    # Separate features and target
    X = df.drop(columns=["fraud_reported"])
    y = df["fraud_reported"]

    # Encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    return model
