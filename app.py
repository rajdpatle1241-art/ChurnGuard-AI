
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnGuard | AI Customer Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "churnguard_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "model_metrics.csv"
CUSTOMER_PATH = BASE_DIR / "artifacts" / "customer_intelligence.csv"

# =========================================================
# 2. PROFESSIONAL DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080F1D;
    color: #EAF2FF;
}

[data-testid="stHeader"] {
    background: rgba(8,15,29,0.95);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #0D1729;
    border-right: 1px solid #1D3150;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF;
    font-weight: 800;
}

[data-testid="stSidebar"] p {
    color: #94A9C8;
}

h1, h2, h3 {
    color: #F1F6FF !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #CBD8EB;
}

.metric-card {
    background: linear-gradient(145deg, #142541, #101D33);
    border: 1px solid #25436B;
    border-radius: 16px;
    padding: 21px;
    min-height: 135px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.metric-label {
    color: #91A8C8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #60A5FA;
    font-size: 12px;
    margin-top: 6px;
}

.hero {
    background: linear-gradient(110deg, #122A4D, #10203A 65%, #152D50);
    border: 1px solid #294C79;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 31px;
    font-weight: 800;
}

.hero-subtitle {
    color: #B5C9E8;
    font-size: 14px;
    margin-top: 9px;
}

.section-card {
    background: #101D32;
    border: 1px solid #223B5E;
    border-radius: 15px;
    padding: 20px;
}

.info-text {
    color: #91A8C8;
    font-size: 13px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #1677F9, #2563EB);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 20px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #398BFF;
    color: white;
    border: 0;
}

[data-testid="stMetric"] {
    background: #13233B;
    border: 1px solid #29466C;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #9DB3D2;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #14233A;
    border-color: #2B466B;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29466C;
    border-radius: 10px;
}

hr {
    border-color: #263D5D;
}

.stProgress > div > div > div {
    background: #2583FF;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. COLORS AND CHART SETTINGS
# =========================================================

BLUE = "#3B82F6"
CYAN = "#22D3EE"
GREEN = "#22C55E"
ORANGE = "#F59E0B"
RED = "#EF4444"

RISK_COLORS = {
    "Low": GREEN,
    "Medium": ORANGE,
    "High": RED
}

CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#101D32",
    "plot_bgcolor": "#101D32",
    "font": {"color": "#DCE8FA", "family": "Inter"},
    "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
    "legend": {"bgcolor": "rgba(0,0,0,0)"}
}

# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_bundle()

# =========================================================
# 5. COMMON FUNCTIONS
# =========================================================

def risk_band(probability):
    if probability < 0.40:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    return "High"


def prepare_input(df):
    cols = bundle["feature_columns"]

    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    x = df[cols].copy()

    if "TotalCharges" in x.columns:
        x["TotalCharges"] = pd.to_numeric(
            x["TotalCharges"], errors="coerce"
        )

    return x


def predict_df(df):
    x = prepare_input(df)

    transformed = bundle["preprocessor"].transform(x)

    probabilities = bundle["models"]["XGBoost"].predict_proba(
        transformed
    )[:, 1]

    output = df.copy()

    output["Churn Probability"] = probabilities
    output["Risk Level"] = [
        risk_band(float(p)) for p in probabilities
    ]

    if "MonthlyCharges" in output.columns:
        output["Estimated Monthly Revenue Exposure"] = (
            pd.to_numeric(
                output["MonthlyCharges"], errors="coerce"
            ).fillna(0) * probabilities
        )

    return output, transformed


def recommendations(row, risk):
    actions = []

    contract = str(row.get("Contract", ""))
    tenure = pd.to_numeric(
        pd.Series([row.get("tenure", 0)]),
        errors="coerce"
    ).fillna(0).iloc[0]

    internet = str(row.get("InternetService", ""))
    security = str(row.get("OnlineSecurity", ""))
    support = str(row.get("TechSupport", ""))
    payment = str(row.get("PaymentMethod", ""))

    if risk == "Low":
        actions.append(
            "Continue standard engagement and monitor account activity."
        )
    else:
        if contract == "Month-to-month":
            actions.append(
                "Offer a suitable long-term contract with transparent savings."
            )

        if tenure < 12:
            actions.append(
                "Provide an onboarding check-in and early-tenure loyalty offer."
            )

        if security == "No" and internet != "No":
            actions.append(
                "Explain available online security options and benefits."
            )

        if support == "No" and internet != "No":
            actions.append(
                "Offer a technical-support check-in or service-quality review."
            )

        if payment == "Electronic check":
            actions.append(
                "Review billing friction and offer convenient payment options."
            )

        if not actions:
            actions.append(
                "Contact the customer to understand service concerns and review plan fit."
            )

    return actions


def kpi(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(fig, height=360):
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(height=height)
    return fig


def show_risk_chart(data):
    counts = (
        data["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Risk Level", "Customers"]

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Customers",
        hole=0.66,
        color="Risk Level",
        color_discrete_map=RISK_COLORS
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#101D32", width=3))
    )

    return style_chart(fig)


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("# 🛡️ ChurnGuard")
    st.caption("AI CUSTOMER INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Customer Prediction",
            "Bulk Analysis",
            "Explainable AI",
            "Retention Actions",
            "Model Performance"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**MODEL STATUS**")
    if bundle is not None:
        st.success("XGBoost model loaded")
    else:
        st.error("Model not available")

    st.caption("IBM Telco Customer Churn dataset")
    st.caption("ChurnGuard • Academic Prototype")

# =========================================================
# 7. MODEL NOT FOUND
# =========================================================

if bundle is None:
    page_header(
        "ChurnGuard",
        "Explainable Customer Churn Prediction & Retention Prioritization"
    )

    st.error("Trained model not found.")

    st.code("python train_model.py", language="powershell")

    st.stop()

# =========================================================
# 8. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    page_header(
        "Customer Intelligence Dashboard",
        "Monitor churn risk, identify priority customers and support retention decisions."
    )

    default_data = None

    uploaded = st.file_uploader(
        "Upload scored customer intelligence CSV (optional)",
        type=["csv"],
        key="overview_upload"
    )

    try:
        if uploaded is not None:
            default_data = pd.read_csv(uploaded)
        elif CUSTOMER_PATH.exists():
            default_data = pd.read_csv(CUSTOMER_PATH)

    except Exception as error:
        st.error(f"Unable to read customer data: {error}")

    if default_data is None:
        st.info(
            "No customer intelligence file found. "
            "Run Bulk Analysis to score your customer dataset."
        )
        st.stop()

    data = default_data

    required_output = ["Risk Level", "Churn Probability"]

    if not all(col in data.columns for col in required_output):
        st.error(
            "This file does not contain churn prediction results. "
            "Upload a scored customer intelligence CSV."
        )
        st.stop()

    total = len(data)
    high = int((data["Risk Level"] == "High").sum())
    medium = int((data["Risk Level"] == "Medium").sum())
    low = int((data["Risk Level"] == "Low").sum())

    exposure = 0

    if "Estimated Monthly Revenue Exposure" in data.columns:
        exposure = data[
            "Estimated Monthly Revenue Exposure"
        ].sum()

    churn_rate = (
        data["Churn Probability"].mean() * 100
        if total > 0 else 0
    )

    st.markdown("### Business snapshot")

    a, b, c, d = st.columns(4)

    with a:
        kpi("CUSTOMERS ANALYSED", f"{total:,}", "Total scored records")

    with b:
        kpi("HIGH-RISK CUSTOMERS", f"{high:,}", "Risk probability ≥ 70%")

    with c:
        kpi("AVERAGE CHURN PROBABILITY", f"{churn_rate:.1f}%", "Mean model probability")

    with d:
        kpi("REVENUE EXPOSURE PROXY", f"{exposure:,.2f}", "Monthly charge × probability")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk distribution")
        st.plotly_chart(
            show_risk_chart(data),
            use_container_width=True
        )

    with right:
        st.subheader("Customer risk volume")

        risk_counts = (
            data["Risk Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Customers"]

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            text="Customers"
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    st.subheader("Priority customer watchlist")

    priority = data.sort_values(
        "Churn Probability", ascending=False
    ).head(15)

    display_cols = [
        col for col in [
            "customerID",
            "tenure",
            "Contract",
            "MonthlyCharges",
            "Churn Probability",
            "Risk Level"
        ] if col in priority.columns
    ]

    st.dataframe(
        priority[display_cols],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Revenue exposure is an estimated probability-weighted monthly "
        "charge proxy, not confirmed revenue loss."
    )


# =========================================================
# 9. CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    page_header(
        "Individual Customer Prediction",
        "Evaluate customer profiles using XGBoost and configurable churn-risk thresholds."
    )

    cols = bundle["feature_columns"]

    # Adjustable high-risk threshold
    st.subheader("Risk Detection Settings")

    risk_threshold = st.slider(
        "High-risk alert threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.70,
        step=0.05,
        format="%.0f%%"
    )

    st.caption(
        f"Customers with a churn probability of {risk_threshold:.0%} "
        "or higher will be classified as High Risk. "
        "Medium Risk starts at 40%."
    )

    defaults = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0
    }

    categorical_options = {
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    }

    st.subheader("Customer account information")

    with st.form("customer_form"):

        formvals = {}

        c1, c2, c3 = st.columns(3)

        for i, col in enumerate(cols):

            container = [c1, c2, c3][i % 3]

            with container:

                if col in categorical_options:

                    options = categorical_options[col]

                    formvals[col] = st.selectbox(
                        col,
                        options,
                        index=options.index(defaults[col])
                        if defaults.get(col) in options else 0
                    )

                elif col == "SeniorCitizen":

                    formvals[col] = st.selectbox(
                        col, [0, 1]
                    )

                elif col == "tenure":

                    formvals[col] = st.number_input(
                        col,
                        min_value=0,
                        max_value=72,
                        value=int(defaults[col])
                    )

                elif col in ["MonthlyCharges", "TotalCharges"]:

                    formvals[col] = st.number_input(
                        col,
                        min_value=0.0,
                        value=float(defaults[col]),
                        step=5.0
                    )

                else:

                    formvals[col] = st.text_input(
                        col,
                        value=str(defaults.get(col, ""))
                    )

        submitted = st.form_submit_button(
            "🛡️ Analyse Customer",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            row = pd.DataFrame([formvals])

            scored, transformed = predict_df(row)

            probability = float(
                scored["Churn Probability"].iloc[0]
            )

            # Apply the selected risk threshold
            if probability >= risk_threshold:
                risk = "High"
            elif probability >= 0.40:
                risk = "Medium"
            else:
                risk = "Low"

            # Store prediction information
            st.session_state["last_customer"] = row
            st.session_state["last_transformed"] = transformed
            st.session_state["last_probability"] = probability
            st.session_state["last_risk"] = risk
            st.session_state["last_threshold"] = risk_threshold

            st.divider()

            st.subheader("Prediction result")

            a, b, c = st.columns(3)

            with a:
                kpi(
                    "CHURN PROBABILITY",
                    f"{probability:.1%}",
                    "XGBoost model output"
                )

            with b:
                kpi(
                    "RISK CLASSIFICATION",
                    risk.upper(),
                    f"High-risk threshold: {risk_threshold:.0%}"
                )

            with c:
                exposure = float(
                    scored[
                        "Estimated Monthly Revenue Exposure"
                    ].iloc[0]
                )

                kpi(
                    "MONTHLY EXPOSURE PROXY",
                    f"{exposure:,.2f}",
                    "Monthly charge × probability"
                )

            st.progress(probability)

            st.subheader("Risk interpretation")

            if risk == "High":

                st.error(
                    "High risk: Prioritise customer engagement."
                )

            elif risk == "Medium":

                st.warning(
                    "Medium risk: Monitor and review service experience."
                )

            else:

                st.success(
                    "Low risk: Continue regular customer engagement."
                )

            st.subheader("Recommended retention actions")

            for action in recommendations(row.iloc[0], risk):
                st.success(action)

            st.info(
                "This prediction is a model estimate, not a certainty "
                "that the customer will leave. Adjusting the threshold "
                "changes risk classification, not the model probability."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# =========================================================
# 10. BULK CUSTOMER ANALYSIS
# =========================================================

elif page == "Bulk Analysis":

    page_header(
        "Bulk Customer Analysis",
        "Score multiple customer records and identify high-priority accounts."
    )

    st.markdown("### Upload customer dataset")

    file = st.file_uploader(
        "Choose a customer CSV file",
        type=["csv"]
    )

    if file is not None:

        try:

            raw = pd.read_csv(file)

            st.metric(
                "Uploaded customer records",
                f"{len(raw):,}"
            )

            st.dataframe(
                raw.head(8),
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "🚀 Run Churn Analysis",
                type="primary"
            ):

                scored, _ = predict_df(raw)

                st.session_state["bulk_scored"] = scored

                st.success(
                    f"Successfully analysed {len(scored):,} customers."
                )

        except Exception as error:
            st.error(f"Unable to analyse file: {error}")

    if "bulk_scored" in st.session_state:

        scored = st.session_state["bulk_scored"]

        st.divider()

        st.subheader("Analysis results")

        high = int((scored["Risk Level"] == "High").sum())
        avg_probability = scored["Churn Probability"].mean()

        a, b, c = st.columns(3)

        with a:
            kpi("CUSTOMERS SCORED", f"{len(scored):,}", "Uploaded records")

        with b:
            kpi("HIGH-RISK ACCOUNTS", f"{high:,}", "Priority segment")

        with c:
            kpi("AVERAGE CHURN RISK", f"{avg_probability:.1%}", "Mean predicted probability")

        st.dataframe(
            scored.sort_values(
                "Churn Probability", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Customer Intelligence CSV",
            scored.to_csv(index=False).encode("utf-8"),
            "churnguard_customer_intelligence.csv",
            "text/csv",
            use_container_width=True
        )

        st.subheader("Risk distribution")

        st.plotly_chart(
            show_risk_chart(scored),
            use_container_width=True
        )


elif page == "Explainable AI":

    page_header(
        "Explainable AI Intelligence",
        "Understand individual predictions and the overall factors influencing customer churn."
    )

    st.info(
        "SHAP explains model behaviour. Feature contributions show "
        "model associations, not proven causes of churn."
    )

    if "last_transformed" not in st.session_state:

        st.warning(
            "First generate a prediction in Customer Prediction, "
            "then return to this page."
        )

    else:

        try:

            model = bundle["models"]["XGBoost"]
            explainer = shap.TreeExplainer(model)

            # =========================================
            # 1. INDIVIDUAL CUSTOMER EXPLANATION
            # =========================================

            st.subheader("Individual Customer Explanation")

            shap_values = explainer.shap_values(
                st.session_state["last_transformed"]
            )

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, 1]
            else:
                values = values.reshape(-1)

            names = bundle["transformed_feature_names"]

            if len(names) != len(values):
                raise ValueError(
                    "Feature names do not match SHAP output dimensions."
                )

            importance = pd.DataFrame({
                "Feature": names,
                "SHAP contribution": values
            })

            importance["Absolute contribution"] = (
                importance["SHAP contribution"].abs()
            )

            top = importance.nlargest(
                12, "Absolute contribution"
            ).sort_values("SHAP contribution")

            st.metric(
                "Predicted churn probability",
                f"{st.session_state['last_probability']:.1%}"
            )

            fig = px.bar(
                top,
                x="SHAP contribution",
                y="Feature",
                orientation="h",
                color="SHAP contribution",
                color_continuous_scale=[
                    "#2563EB", "#60A5FA", "#F59E0B", "#EF4444"
                ]
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="white"
            )

            st.plotly_chart(
                style_chart(fig, 500),
                use_container_width=True
            )

            st.caption(
                "Positive contributions push the model output toward "
                "churn; negative contributions push it away. Values "
                "may be on the raw model output scale, not probability."
            )

            st.dataframe(
                top[[
                    "Feature",
                    "SHAP contribution",
                    "Absolute contribution"
                ]].sort_values(
                    "Absolute contribution", ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            # =========================================
            # 2. GLOBAL FEATURE IMPORTANCE
            # =========================================

            st.divider()

            st.subheader("Global Churn Drivers")

            st.write(
                "This analysis measures the average absolute SHAP "
                "contribution of each feature across multiple customers."
            )

            try:

                data_path = (
                    Path(__file__).resolve().parent
                    / "artifacts"
                    / "customer_intelligence.csv"
                )

                if not data_path.exists():
                    raise FileNotFoundError(
                        "customer_intelligence.csv was not found."
                    )

                global_data = pd.read_csv(data_path)

                feature_columns = bundle["feature_columns"]

                missing = [
                    col for col in feature_columns
                    if col not in global_data.columns
                ]

                if missing:
                    raise ValueError(
                        f"Missing required features: {missing}"
                    )

                # Use a representative sample for faster SHAP analysis
                sample_size = min(100, len(global_data))

                global_sample = global_data[
                    feature_columns
                ].sample(
                    n=sample_size,
                    random_state=42
                )

                # Apply the same preprocessing used during training
                preprocessor = bundle["preprocessor"]

                transformed_global = preprocessor.transform(
                    global_sample
                )

                # Convert sparse matrices to dense when necessary
                if hasattr(transformed_global, "toarray"):
                    transformed_global = transformed_global.toarray()

                global_shap = explainer.shap_values(
                    transformed_global
                )

                if isinstance(global_shap, list):
                    global_shap = global_shap[-1]

                global_values = np.asarray(global_shap)

                if global_values.ndim == 3:
                    global_values = global_values[:, :, 1]

                global_names = bundle["transformed_feature_names"]

                if global_values.shape[1] != len(global_names):
                    raise ValueError(
                        "Global SHAP feature dimensions do not match."
                    )

                # Mean absolute SHAP contribution across customers
                global_importance = pd.DataFrame({
                    "Feature": global_names,
                    "Mean Absolute SHAP": np.mean(
                        np.abs(global_values), axis=0
                    )
                })

                global_importance = global_importance.sort_values(
                    "Mean Absolute SHAP",
                    ascending=False
                ).head(12)

                st.metric(
                    "Customers analysed",
                    f"{sample_size:,}"
                )

                global_fig = px.bar(
                    global_importance.sort_values(
                        "Mean Absolute SHAP"
                    ),
                    x="Mean Absolute SHAP",
                    y="Feature",
                    orientation="h",
                    color="Mean Absolute SHAP",
                    color_continuous_scale=[
                        "#1E3A8A", "#2563EB", "#60A5FA"
                    ],
                    title="Top 12 Global Churn Influencing Features"
                )

                st.plotly_chart(
                    style_chart(global_fig, 500),
                    use_container_width=True
                )

                st.dataframe(
                    global_importance,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Higher mean absolute SHAP values indicate greater "
                    "average influence on model output. They do not "
                    "indicate whether a feature increases or decreases "
                    "churn, and do not establish causation."
                )

            except Exception as global_error:

                st.warning(
                    f"Global SHAP analysis unavailable: {global_error}"
                )

        except Exception as error:

            st.error(
                f"SHAP explanation could not be generated: {error}"
            )



# =========================================================
# 12. RETENTION ACTIONS
# =========================================================

elif page == "Retention Actions":

    page_header(
        "Retention Intelligence",
        "Translate customer risk predictions into practical retention actions."
    )

    if "last_customer" not in st.session_state:

        st.info(
            "Generate a prediction in Customer Prediction "
            "to see personalised retention actions."
        )

    else:

        row = st.session_state["last_customer"].iloc[0]

        probability = st.session_state["last_probability"]

        risk = st.session_state.get(
            "last_risk",
            risk_band(probability)
        )

        st.subheader("Customer risk profile")

        a, b = st.columns(2)

        with a:
            kpi(
                "RISK LEVEL",
                risk.upper(),
                "Predicted customer risk"
            )

        with b:
            kpi(
                "CHURN PROBABILITY",
                f"{probability:.1%}",
                "XGBoost model output"
            )

        st.divider()

        st.subheader("Recommended action plan")

        actions = recommendations(row, risk)

        for i, action in enumerate(actions, 1):

            st.markdown(
                f"""
                <div class="section-card">
                    <span style="color:#60A5FA;font-weight:800;">
                        ACTION {i}
                    </span>
                    <br><br>
                    <span style="color:#F1F6FF;">
                        {action}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

        st.caption(
            "These recommendations are rule-based decision-support "
            "suggestions. Their impact on actual retention has not "
            "been experimentally validated."
        )


# =========================================================
# 13. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    page_header(
        "Model Performance & Evaluation",
        "Compare machine-learning models using held-out test-set metrics."
    )

    if not METRICS_PATH.exists():

        st.warning(
            "Model metrics file not found. Run python train_model.py."
        )

    else:

        metrics = pd.read_csv(METRICS_PATH)

        st.subheader("Model comparison")

        numeric_cols = [
            col for col in metrics.columns
            if col != "Model"
        ]

        st.dataframe(
            metrics.style.format({
                col: "{:.2%}" for col in numeric_cols
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        metric = st.selectbox(
            "Select evaluation metric",
            numeric_cols,
            index=numeric_cols.index("Accuracy")
            if "Accuracy" in numeric_cols else 0
        )

        fig = px.bar(
            metrics,
            x="Model",
            y=metric,
            color="Model",
            text=metric,
            title=f"{metric} comparison"
        )

        fig.update_traces(
            texttemplate="%{text:.2%}",
            textposition="outside"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

        if "ROC-AUC" in metrics.columns:

            st.subheader("ROC-AUC comparison")

            roc = px.bar(
                metrics,
                x="Model",
                y="ROC-AUC",
                color="Model",
                text="ROC-AUC"
            )

            roc.update_traces(
                texttemplate="%{text:.2%}",
                textposition="outside"
            )

            st.plotly_chart(
                style_chart(roc),
                use_container_width=True
            )

        if "XGBoost" in metrics["Model"].values:

            xgb = metrics[
                metrics["Model"] == "XGBoost"
            ].iloc[0]

            st.subheader("XGBoost evaluation")

            a, b, c, d = st.columns(4)

            for container, metric_name in zip(
                [a, b, c, d],
                ["Accuracy", "Precision", "Recall", "F1-score"]
            ):

                with container:
                    kpi(
                        metric_name.upper(),
                        f"{xgb[metric_name]:.2%}",
                        "Held-out test result"
                    )

        st.caption(
            "Metrics are calculated on the held-out test partition. "
            "Performance is specific to the dataset, preprocessing "
            "and evaluation setup used during training."
        )

# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#7186A6;font-size:12px;">
        🛡️ <b>CHURNGUARD</b> &nbsp; | &nbsp;
        AI-Powered Customer Retention Intelligence
        <br><br>
        Academic Project Prototype • IBM Telco Customer Churn Dataset
    </div>
    """,
    unsafe_allow_html=True
)
