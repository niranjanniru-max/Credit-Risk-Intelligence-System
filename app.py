import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    model   = joblib.load('credit_risk_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_model()

FEATURE_IMPORTANCE = {
    'credit_type':        0.19,
    'income':             0.16,
    'loan_amount':        0.14,
    'property_value':     0.14,
    'Credit_Score':       0.14,
    'term':               0.02,
    'Neg_ammortization':  0.01,
    'lump_sum_payment':   0.01,
    'occupancy_type':     0.01,
    'loan_purpose':       0.01,
}

st.sidebar.title("🏦 Credit Risk System")
st.sidebar.markdown("---")
st.sidebar.subheader("Applicant Details")
st.sidebar.caption("💡 High Risk: CRIF, Score=520, Income=1500, lpsm")
st.sidebar.caption("💡 Low Risk:  CIB,  Score=820, Income=15000, not_lpsm")
st.sidebar.markdown("---")

loan_amount    = st.sidebar.number_input("Loan Amount ($)",      min_value=0.0,   value=200000.0, step=1000.0)
income         = st.sidebar.number_input("Monthly Income ($)",   min_value=0.0,   value=5000.0,   step=100.0)
Credit_Score   = st.sidebar.number_input("Credit Score",         min_value=300.0, max_value=850.0, value=650.0, step=1.0)
property_value = st.sidebar.number_input("Property Value ($)",   min_value=0.0,   value=300000.0, step=1000.0)
term           = st.sidebar.number_input("Loan Term (months)",   min_value=0.0,   value=360.0,    step=12.0)
st.sidebar.markdown("---")
credit_type       = st.sidebar.selectbox("Credit Type",           ['CIB', 'EXP', 'CRIF', 'EQUI'])
loan_type         = st.sidebar.selectbox("Loan Type",             ['type1', 'type2', 'type3'])
loan_purpose      = st.sidebar.selectbox("Loan Purpose",          ['p1', 'p2', 'p3', 'p4'])
lump_sum_payment  = st.sidebar.selectbox("Lump Sum Payment",      ['not_lpsm', 'lpsm'])
Neg_ammortization = st.sidebar.selectbox("Negative Amortization", ['not_neg', 'neg_amm'])
occupancy_type    = st.sidebar.selectbox("Occupancy Type",        ['pr', 'sr', 'ir'])
Credit_Worthiness = st.sidebar.selectbox("Credit Worthiness",     ['l1', 'l2'])
open_credit       = st.sidebar.selectbox("Open Credit",           ['nopc', 'opc'])
interest_only     = st.sidebar.selectbox("Interest Only",         ['not_int', 'int_only'])
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Default Risk", type="primary", use_container_width=True)

st.title("🏦 Credit Risk Intelligence System")
st.caption("Built with Random Forest · 86% Accuracy · 148,670 loan records trained")
st.markdown("---")

if predict_btn:
    input_data = pd.DataFrame([{
        'loan_type':          loan_type,
        'loan_purpose':       loan_purpose,
        'Credit_Worthiness':  Credit_Worthiness,
        'open_credit':        open_credit,
        'loan_amount':        loan_amount,
        'term':               term,
        'Neg_ammortization':  Neg_ammortization,
        'interest_only':      interest_only,
        'lump_sum_payment':   lump_sum_payment,
        'property_value':     property_value,
        'occupancy_type':     occupancy_type,
        'income':             income,
        'credit_type':        credit_type,
        'Credit_Score':       Credit_Score,
    }])

    input_data  = input_data[model_columns]
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    prob_pct    = round(probability * 100, 1)

    if probability < 0.40:
        risk_label = "Low Risk"
        risk_color = "green"
        emoji      = "🟢"
        bar_color  = "#27AE60"
    elif probability < 0.65:
        risk_label = "Medium Risk"
        risk_color = "orange"
        emoji      = "🟡"
        bar_color  = "#F0A500"
    else:
        risk_label = "High Risk"
        risk_color = "red"
        emoji      = "🔴"
        bar_color  = "#E24B4A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Default Probability", f"{prob_pct}%")
    col2.metric("Risk Category",       risk_label)
    col3.metric("Model Accuracy",      "86%")
    col4.metric("ROC-AUC Score",       "0.84")
    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"{emoji} Risk Assessment")
        st.markdown(f"**:{risk_color}[{risk_label}]** — {prob_pct}% chance of default")
        st.progress(probability)

        if prediction == 1:
            st.error("**Recommendation: REJECT** — High probability of defaulting. Consider requiring additional collateral or a co-signer.")
        elif probability >= 0.40:
            st.warning("**Recommendation: REVIEW** — Moderate risk. Request additional documentation before approving.")
        else:
            st.success("**Recommendation: APPROVE** — Applicant appears financially stable with low default risk.")

        st.markdown("---")
        st.subheader("📊 Risk Gauge")

        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': bar_color},
                'steps': [
                    {'range': [0,  40], 'color': '#EAF3DE'},
                    {'range': [40, 65], 'color': '#FAEEDA'},
                    {'range': [65, 100],'color': '#FCEBEB'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': prob_pct
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🔍 Top 5 Risk Factors")
        val_map = {
            'credit_type':        credit_type,
            'income':             f"${income:,.0f}",
            'loan_amount':        f"${loan_amount:,.0f}",
            'property_value':     f"${property_value:,.0f}",
            'Credit_Score':       str(int(Credit_Score)),
            'term':               f"{int(term)} months",
            'Neg_ammortization':  Neg_ammortization,
            'lump_sum_payment':   lump_sum_payment,
            'occupancy_type':     occupancy_type,
            'loan_purpose':       loan_purpose,
        }
        icons = ["🔴","🟠","🟡","🔵","⚪"]
        top5 = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feat, imp) in enumerate(top5):
            user_val = val_map.get(feat, '')
            st.markdown(f"{icons[i]} **{i+1}. {feat}** → `{user_val}`")
            st.progress(imp)
            st.caption(f"Importance: {imp*100:.1f}%")

        st.markdown("---")
        st.subheader("📋 Applicant Summary")
        summary = pd.DataFrame({
            'Field': ['Loan Amount','Income','Credit Score','Property Value','Credit Type','Lump Sum'],
            'Value': [f"${loan_amount:,.0f}", f"${income:,.0f}", str(int(Credit_Score)),
                      f"${property_value:,.0f}", credit_type, lump_sum_payment]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Full Feature Importance")
    imp_df = pd.DataFrame(
        list(FEATURE_IMPORTANCE.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)

    import plotly.express as px
    fig2 = px.bar(
        imp_df, x='Importance', y='Feature', orientation='h',
        text=imp_df['Importance'].apply(lambda x: f"{x*100:.1f}%"),
        color='Importance',
        color_continuous_scale=['#EAF3DE', '#27AE60', '#1A5276'],
    )
    fig2.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=40,t=20,b=0), height=350)
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🔧 View raw input sent to model"):
        st.dataframe(input_data, use_container_width=True)

else:
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1**\nEnter applicant details in the sidebar")
    col2.info("**Step 2**\nClick Predict Default Risk")
    col3.info("**Step 3**\nGet risk score + top factors")
    st.markdown("---")
    st.subheader("ℹ️ About This System")
    st.markdown("""
    Predicts whether a loan applicant is likely to **default** on their loan.
    Trained on **148,670 real loan records** using a Random Forest classifier.

    **Key findings during development:**
    - Discovered and removed **data leakage** in 3 columns
    - Found **EQUI credit bureau** customers had 99.9% historical default rate
    - Used **IterativeImputer** for smart null handling
    - Applied **SMOTE** to handle class imbalance (75% safe vs 25% default)
    - Tuned with **GridSearchCV** to find optimal hyperparameters
    """)
    st.markdown("---")
    st.subheader("📊 What each field means")
    st.table(pd.DataFrame({
        'Field':   ['Loan Amount','Monthly Income','Credit Score','Property Value',
                    'Term','Credit Type','Lump Sum Payment','Neg. Amortization',
                    'Occupancy Type','Credit Worthiness'],
        'Meaning': [
            'Total loan amount requested ($)',
            'Applicant monthly income ($)',
            'Credit bureau score (300-850, higher = better)',
            'Market value of collateral property ($)',
            'Loan duration in months (360 = 30 years)',
            'Credit bureau used — CIB/EXP/CRIF/EQUI',
            'Whether loan has lump sum payment option',
            'Whether loan allows negative amortization',
            'Primary (pr) / Secondary (sr) / Investment (ir)',
            'Internal credit worthiness rating (l1/l2)'
        ]
    }))

