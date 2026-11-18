import os, json, warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
os.makedirs("artifacts", exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Download the IBM Telco Customer Churn CSV and place it in the data folder."
    )

df = pd.read_csv(DATA_PATH)
if "Churn" not in df.columns:
    raise ValueError("Dataset must contain the target column 'Churn'.")
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["Churn"]).copy()

y = df["Churn"].map({"No":0, "Yes":1})
X = df.drop(columns=["Churn"])
feature_columns = list(X.columns)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

# As documented in the project paper, oversampling is applied to training data only.
# SMOTE after one-hot encoding may create fractional encoded values; this limitation
# should be acknowledged during the jury discussion.
try:
    sampler = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sampler.fit_resample(X_train_t, y_train)
except Exception as exc:
    print(f"SMOTE could not be applied ({exc}); proceeding with original training distribution.")
    X_train_bal, y_train_bal = X_train_t, y_train

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
except Exception as exc:
    raise RuntimeError("XGBoost is required. Install dependencies using: pip install -r requirements.txt") from exc

metric_rows = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_bal, y_train_bal)
    prob = model.predict_proba(X_test_t)[:,1]
    pred = (prob >= 0.5).astype(int)
    metric_rows.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,pred),
        "Precision":precision_score(y_test,pred,zero_division=0),
        "Recall":recall_score(y_test,pred,zero_division=0),
        "F1-score":f1_score(y_test,pred,zero_division=0),
        "ROC-AUC":roc_auc_score(y_test,prob)
    })

metrics = pd.DataFrame(metric_rows)
metrics.to_csv("artifacts/model_metrics.csv", index=False)
# Integrated project workflow selects XGBoost, as specified in the paper.
bundle = {
    "preprocessor":preprocessor,
    "models":models,
    "feature_columns":feature_columns,
    "numeric_columns":numeric_cols,
    "categorical_columns":categorical_cols,
    "transformed_feature_names":preprocessor.get_feature_names_out().tolist()
}
joblib.dump(bundle, "artifacts/churnguard_model.joblib")
with open("artifacts/feature_columns.json","w") as f:
    json.dump(feature_columns,f,indent=2)

# Score the held-out test customers and save a real example intelligence file.
xgb = models["XGBoost"]
test_prob = xgb.predict_proba(X_test_t)[:,1]
scored = X_test.copy()
scored["Churn Probability"] = test_prob
scored["Risk Level"] = pd.cut(
    test_prob, bins=[-0.000001,0.40,0.70,1.000001],
    labels=["Low","Medium","High"], right=False
).astype(str)
scored["Estimated Monthly Revenue Exposure"] = (
    pd.to_numeric(scored["MonthlyCharges"], errors="coerce").fillna(0) * test_prob
)
scored.to_csv("artifacts/customer_intelligence.csv", index=False)

print("\nTraining complete. Metrics on held-out test data:")
print(metrics.to_string(index=False))
print("\nCreated:")
print("- artifacts/churnguard_model.joblib")
print("- artifacts/model_metrics.csv")
print("- artifacts/customer_intelligence.csv")
