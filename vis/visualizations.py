import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visuals(processed_data_path: str):
    """
    Generates and saves basic visualizations for fraud detection insights.
    """
    df = pd.read_csv(processed_data_path)

    # Fraud Count
    plt.figure(figsize=(6, 4))
    sns.countplot(x="fraud_reported", data=df)
    plt.title("Fraud Reported Count")
    plt.xlabel("Fraud Reported")
    plt.ylabel("Count")
    plt.savefig("data/outputs/fraud_count.png")
    plt.close()

    # Fraud by Incident Type
    plt.figure(figsize=(10, 5))
    sns.countplot(x="incident_type", hue="fraud_reported", data=df)
    plt.title("Fraud by Incident Type")
    plt.xlabel("Incident Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/outputs/fraud_by_incident_type.png")
    plt.close()

    # Total Claim Amount by Fraud
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="fraud_reported", y="total_claim_amount", data=df)
    plt.title("Total Claim Amount by Fraud Status")
    plt.xlabel("Fraud Reported")
    plt.ylabel("Total Claim Amount")
    plt.tight_layout()
    plt.savefig("data/outputs/fraud_claim_amount.png")
    plt.close()

    print("[INFO] Visualizations saved to data/outputs/")
