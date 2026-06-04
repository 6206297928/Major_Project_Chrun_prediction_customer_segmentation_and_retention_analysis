import streamlit as st
import pandas as pd
import numpy as np
import requests

# ─── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FastAPI Engine Link
FASTAPI_URL = st.secrets.get("FASTAPI_URL", "https://your-render-app-name.onrender.com/predict")

# ─── CLEAN PANEL UI CSS ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Global structural overrides */
    .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* Executive Metric Container Layouts */
    .metric-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-panel p { margin: 0; font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }

    /* Risk Strategy Banner Components */
    .action-banner {
        padding: 1.25rem;
        border-radius: 6px;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        font-weight: 500;
    }
    .banner-danger { background-color: #FEF2F2; color: #991B1B; border-left: 4px solid #EF4444; }
    .banner-success { background-color: #ECFDF5; color: #065F46; border-left: 4px solid #10B981; }

    /* Structural Section Header Layouts */
    .ui-section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Explicit Operational Playbooks (Mapped to Churn Risk Output)
RETENTION_PLAYBOOK = {
    "High Risk": [
        "Initiate a contract migration campaign offering a 25% rate reduction contingent on a 12-month commitment.",
        "Deploy a complimentary 6-month system security layer (Online Security / Tech Support bundle).",
        "Automate billing transition from manual electronic check to authorized direct credit card clearing."
    ],
    "Stable": [
        "Enroll account in standard tier automated retention cycle.",
        "Deliver annual utility statements highlighting core infrastructure savings metrics."
    ]
}

# Plain-English Customer Summaries (Non-Technical)
SEGMENT_DETAILS = {
    "High-Value Loyal": "Pays a premium, uses multiple services, and has been with us long-term.",
    "High-Risk High-Cost": "A brand new customer paying a high bill on a month-to-month plan; high risk of leaving.",
    "Low-Usage Stable": "Has basic, low-cost services but stays with us reliably over time.",
    "Mid-Tier Transitional": "An average, standard account currently approaching a contract renewal window."
}

# Rule-Based Segmentation Engine Matrix
def compute_customer_segment(contract: str, tenure: int, monthly: float) -> str:
    if contract != "Month-to-month" and tenure > 24 and monthly > 75:
        return "High-Value Loyal"
    elif contract == "Month-to-month" and tenure <= 12:
        return "High-Risk High-Cost"
    elif monthly <= 45 and tenure > 12:
        return "Low-Usage Stable"
    else:
        return "Mid-Tier Transitional"

# ─── SIDEBAR DATA CAPTURE CONTROL SYSTEMS ──────────────────────────────────────
with st.sidebar:
    st.markdown("### Customer Parameters")
    st.markdown("Configure variables to compute real-time operational risk matrices.")
    st.markdown("---")

    with st.expander("Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner Dependency", ["Yes", "No"])
        dependents = st.selectbox("Family Dependents", ["Yes", "No"])

    with st.expander("Account Configuration", expanded=True):
        tenure = st.slider("Account Lifespan (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing Registration", ["Yes", "No"])
        payment = st.selectbox("Payment Gateway Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    with st.expander("Core Service Allocations", expanded=False):
        phone = st.selectbox("Voice Architecture", ["Yes", "No"])
        multi_lines = st.selectbox("Multi-Line Array", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Broadband Category", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security Infrastructure", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Cloud Vault Backup Services", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Hardware Device Protection Coverage", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Dedicated Enterprise Technical Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("IPTV Streaming Architecture", ["Yes", "No", "No internet service"])
        movies = st.selectbox("On-Demand Media Streaming Assets", ["Yes", "No", "No internet service"])

    with st.expander("Financial Allocation Specs", expanded=True):
        monthly = st.number_input("Monthly Operating Charge ($)", 18.0, 120.0, 70.0, step=5.0)
        total = st.number_input("Total Cumulative Billings ($)", 0.0, 9000.0, monthly * tenure, step=50.0)

    st.markdown("---")
    evaluate_engine_trigger = st.button("Compute Churn Variance", type="primary", use_container_width=True)

# ─── EXECUTIVE DASHBOARD DISPLAY INTERFACE ─────────────────────────────────────
st.markdown("## Customer Risk Assessment Pipeline")
st.markdown("Inference client utilizing an optimized multi-tier Artificial Neural Network backend.")
st.markdown("---")

if evaluate_engine_trigger:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total)
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            server_data = response.json()
            raw_prob = server_data["ann_churn_probability"]
            label_text = server_data["ann_prediction_text"]

            assigned_segment = compute_customer_segment(contract, tenure, monthly)
            segment_quote = SEGMENT_DETAILS.get(assigned_segment, "")

            # --- Executive Core Performance Panels (Strict Custom Sizing Overrides) ---
            st.markdown('<div class="ui-section-header">Performance Metrics</div>', unsafe_allow_html=True)
            col_1, col_2, col_3 = st.columns(3)

            with col_1:
                classification_text = "Risk Detected" if label_text == "Yes" else "Stable Account"
                st.markdown(
                    f'''<div class="metric-panel">
                        <p>Classification Result</p>
                        <div style="font-size: 24px; font-weight: 700; color: #1E293B; margin-top: 0.5rem;">{classification_text}</div>
                    </div>''',
                    unsafe_allow_html=True
                )
            with col_2:
                st.markdown(
                    f'''<div class="metric-panel">
                        <p>Model Probability</p>
                        <div style="font-size: 24px; font-weight: 700; color: #1E293B; margin-top: 0.5rem;">{raw_prob:.2%}</div>
                    </div>''',
                    unsafe_allow_html=True
                )
            with col_3:
                st.markdown(
                    f'''<div class="metric-panel">
                        <p>Assigned Segment Cohort</p>
                        <div style="font-size: 24px; font-weight: 700; color: #1E293B; margin-top: 0.5rem;">{assigned_segment}</div>
                        <div style="font-size: 13px; color: #64748B; font-weight: 500; font-style: italic; margin-top: 0.4rem; padding: 0 5px; line-height: 1.3;">"{segment_quote}"</div>
                    </div>''',
                    unsafe_allow_html=True
                )

            # --- Mitigation Protocols Layout ---
            st.markdown('<div class="ui-section-header">Retention Execution Strategy</div>', unsafe_allow_html=True)
            playbook_index = "High Risk" if label_text == "Yes" else "Stable"
            banner_theme = "banner-danger" if label_text == "Yes" else "banner-success"

            for action in RETENTION_PLAYBOOK[playbook_index]:
                st.markdown(f'<div class="action-banner {banner_theme}">{action}</div>', unsafe_allow_html=True)

        else:
            st.error(f"Backend Server Failure Exception. Server responded with status code: {response.status_code}")

    except Exception as network_exception:
        st.error(f"Network transport pipeline lost connection to FastAPI at port 8000: {str(network_exception)}")

else:
    st.info("Input profile vectors via control panels and trigger calculation matrix to display analytics.")
