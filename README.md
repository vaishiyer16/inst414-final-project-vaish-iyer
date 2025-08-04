# Insurance Claim Fraud Detection

## Project Overview

This project aims to detect **fraudulent insurance claims** using machine learning. Fraudulent claims pose a significant financial burden on insurance companies, and effective fraud detection models can help prioritize suspicious claims for review.

### Business Problem
Fraudulent insurance claims reduce the operational efficiency and profitability of insurers. This project supports the **claims department at an insurance company** (like Chubb or Allstate) in identifying which claims are most likely to be fraudulent using a predictive model trained on historical claim data.

### Dataset Used
- **`insurance_claims.csv`**
  - Contains ~1,000 historical insurance claim records.
  - Each record includes customer details, policy info, incident details, and a fraud flag.

###  Techniques Employed
- ETL (Extract → Transform → Load) Pipeline
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Classification (Random Forest)
- Performance Evaluation (Precision, Recall, F1-score)
- Visualizations (via Matplotlib)

###  Expected Outputs
- A cleaned dataset stored for modeling
- A trained Random Forest classifier
- Evaluation metrics on model performance
- Visualizations of class imbalance and fraud prediction accuracy

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vaishiyer16/inst414-final-project-vaish-iyer.git
   cd inst414-final-project-vaish-iyer

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
The project is executed through the main.py file, which runs all stages of the data science pipeline:

bash
Copy
Edit
python main.py
During execution, you will see log messages indicating:

Data extraction status

Data shape after transformation

Model training results

Visual output generation

Outputs
Raw extracted data: data/extracted/insurance_claims_raw.csv

Cleaned data: data/processed/insurance_claims_cleaned.csv

Visuals: data/outputs/

inst414-final-project-vaish-iyer/
│
├── data/
│   ├── extracted/              # Raw data
│   ├── processed/              # Cleaned data
│   ├── outputs/                # Charts and plots
│   ├── reference-tables/      # Data dictionaries, lookup tables
│
├── etl/
│   ├── extract.py              # Data loading logic
│   ├── transform.py            # Cleaning & preprocessing
│   ├── load.py                 # Save/load utility
│
├── analysis/
│   ├── model.py                # Model training & evaluation
│
├── vis/
│   ├── visualizations.py       # Fraud detection visuals
│
├── main.py                     # Main project execution script
├── requirements.txt            # Installed libraries
├── README.md                   # Project documentation

