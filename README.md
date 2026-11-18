# ChurnGuard — Explainable Customer Churn Prediction

A Streamlit prototype for customer churn prediction, risk categorization, SHAP explanations, bulk scoring and rule-based retention suggestions.

## 1. Project requirements

- Python 3.10 or 3.11 recommended
- VS Code
- IBM Telco Customer Churn CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

The project uses the same general workflow described in the ChurnGuard project report and paper: preprocessing, Logistic Regression / Random Forest / XGBoost comparison, training-only SMOTE, XGBoost scoring, SHAP explanation and rule-based recommendations.

## 2. Get the dataset

Use the IBM Telco Customer Churn sample dataset (7,043 rows, 21 original columns). Download the CSV from the [Kaggle IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Select the file named `WA_Fn-UseC_-Telco-Customer-Churn.csv`. The file must be named:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place it here:

`ChurnGuard_Project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Do not upload a different dataset unless it has the same feature columns.

## 3. Open in VS Code

1. Extract the ZIP.
2. Open the `ChurnGuard_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Train the ML models

After placing the CSV in `data/`, run:

```bash
python train_model.py
```

This trains three classifiers and creates:
- `artifacts/churnguard_model.joblib`
- `artifacts/model_metrics.csv`
- `artifacts/customer_intelligence.csv`

Training may take a few minutes. Do not report metrics from the PPT or paper as the newly trained model's metrics; use the metrics file generated on your machine.

## 5. Run the website

```bash
streamlit run app.py
```

The browser will open a local website, usually at `http://localhost:8501`.

## 6. Website modules

- Overview: KPIs and risk distribution from a scored CSV.
- Customer Prediction: enter customer attributes and run XGBoost prediction.
- Bulk Analysis: upload a customer CSV, score rows and download results.
- Explainable AI: local SHAP feature contribution plot for the latest individual prediction.
- Retention Actions: transparent rule-based suggestions.
- Model Performance: compare the three models using held-out test metrics.

## 7. Google Colab option

Upload the project ZIP and dataset to Colab, extract the ZIP, change into the project folder, then run:

```python
%cd /content/ChurnGuard_Project
!pip install -r requirements.txt
!python train_model.py
!streamlit run app.py &>/content/streamlit.log &
```

For a public jury demo, run locally on your laptop. Colab's local Streamlit URL is not automatically a public website.

## 8. Important project limitations to state honestly

- This is a prototype trained on the public IBM Telco sample dataset, not a live telecom CRM.
- Risk thresholds are project-defined: below 0.40 Low, 0.40 to below 0.70 Medium, and 0.70 or above High.
- SHAP explains model behaviour; it does not prove that a feature causes churn.
- Retention actions are rule-based suggestions and have not been validated by a controlled intervention.
- Revenue exposure is a probability-weighted monthly-charge proxy, not actual saved revenue or customer lifetime value.
- SMOTE is applied only to training data. Applied after one-hot encoding, it can generate fractional encoded values; this is a methodological limitation to acknowledge.

## 9. Quick troubleshooting

**Dataset not found:** check the exact filename and `data/` folder.

**XGBoost install error:** use Python 3.10/3.11 in a fresh virtual environment and rerun `pip install -r requirements.txt`.

**Streamlit says model missing:** run `python train_model.py` successfully before starting the app.

**Bulk upload says columns missing:** use the original Telco CSV columns; remove `Churn` if present is fine, but do not rename feature columns.

**SHAP error:** confirm `shap` installed correctly. The core prediction app can still run even if the SHAP page has a package/version issue.
