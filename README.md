Insurance Claim Fraud Detection
Project Overview
This project aims to detect fraudulent insurance claims using machine learning. Fraudulent claims pose a significant financial burden on insurance companies, and effective fraud detection models can help prioritize suspicious claims for review.

Business Problem
Fraudulent insurance claims reduce the operational efficiency and profitability of insurers. This project supports the claims department at an insurance company (like Chubb or Allstate) in identifying which claims are most likely to be fraudulent using a predictive model trained on historical claim data.

Dataset Used
insurance_claims.csv

Contains ~1,000 historical insurance claim records.

Each record includes customer details, policy info, incident details, and a fraud flag.

data_dictionary_insurance_claims.csv

Provides definitions for all variables in the dataset (stored in data/reference-tables/).

📊 Techniques Employed
End-to-End ETL (Extract → Transform → Load) Pipeline

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Machine Learning Classification

Logistic Regression (baseline)

Random Forest Classifier (main model)

Model Evaluation & Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC

Confusion matrices and ROC curves

Logging & Error Handling

Centralized logger in main.py

logs/pipeline.log for pipeline run history

try/except blocks around ETL, modeling, and visualization steps

Visualizations

Class imbalance charts

Confusion matrices

ROC curves

📦 Expected & Generated Outputs
When running the pipeline, the following outputs are created:

Raw extracted data → data/extracted/insurance_claims_raw.csv

Cleaned data → data/processed/insurance_claims_cleaned.csv

Evaluation artifacts → data/evaluations/

logisticregression_report.json

randomforest_report.json

metrics_summary.csv

plots/ (confusion matrices, ROC curves)

Visuals → data/outputs/ (EDA and feature insights)

Logs → logs/pipeline.log

Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/vaishiyer16/inst414-final-project-vaish-iyer.git
cd inst414-final-project-vaish-iyer
Create and Activate a Virtual Environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Project

bash
Copy
Edit
python main.py
During execution, you will see log messages indicating:

Data extraction status

Data shape after transformation

Model training and evaluation

Visualization generation

inst414-final-project-vaish-iyer/
│
├── data/
│   ├── extracted/              # Raw data
│   ├── processed/              # Cleaned data
│   ├── outputs/                # EDA charts and plots
│   ├── evaluations/            # Model metrics & evaluation plots
│   │   ├── plots/               # Confusion matrices, ROC curves
│   ├── reference-tables/       # Data dictionaries, lookup tables
│
├── etl/
│   ├── extract.py               # Data loading logic
│   ├── transform.py             # Cleaning & preprocessing
│   ├── load.py                  # Save/load utility
│
├── analysis/
│   ├── model.py                 # Model training & evaluation
│
├── vis/
│   ├── visualizations.py        # Fraud detection visuals
│
├── logs/
│   ├── pipeline.log             # Run history and debug info
│
├── main.py                      # Main project execution script
├── requirements.txt             # Installed libraries
├── README.md                    # Project documentation


Part 3 updated version of read me