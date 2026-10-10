import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #101714; color: #e7eee9; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { font-family: Georgia, serif; font-weight: 500; color: #f1f5f1; }
    h2, h3 { color: #dce8e0; }
    [data-testid="stWidgetLabel"] *, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #dce7df !important;
    }
    [data-testid="stNumberInput"] input, [data-baseweb="base-input"],
    [data-baseweb="input"], [data-baseweb="select"] > div {
        background: #1b2721 !important; color: #eef3ef !important; border-color: #43544a !important;
    }
    [data-baseweb="select"] *, [data-testid="stSlider"] * { color: #dce7df; }
    .eyebrow { color: #e7a27c; font: 700 0.76rem/1.4 monospace; letter-spacing: 0.08em; }
    .intro { max-width: 720px; color: #a8b7ae; font-size: 1rem; }
    .confusion-matrix { width: 100%; border-collapse: collapse; background: #1a2520; color: #ecf2ed; }
    .confusion-matrix th, .confusion-matrix td { border: 1px solid #39483f; padding: 0.65rem 0.8rem; text-align: left; }
    .confusion-matrix th { background: #26362e; font-weight: 650; }
    div[data-testid="stMetric"] { background: #1a2520; border: 1px solid #39483f; padding: 0.9rem 1rem; border-radius: 6px; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: #28795e; color: #fff; border: 0; border-radius: 4px;
        min-height: 2.8rem; font-weight: 650;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #34916e; color: #fff; border: 0;
    }
    div[data-testid="stProgress"] > div > div { background: #e7a27c; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts(artifact_signature):
    scaler_path = os.path.join("models", "scaler.pkl")
    model_path = os.path.join("models", "model.pkl")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.error("Model files not found. Run `src/cleaned_data.py` and `src/train.py` first.")
        st.stop()

    return joblib.load(scaler_path), joblib.load(model_path)


@st.cache_data
def get_validation_metrics(artifact_signature):
    data_path = os.path.join("data", "processed", "cleaned_churn_data.csv")
    if not os.path.exists(data_path):
        return None

    data = pd.read_csv(data_path)
    features = data.drop(columns=["churn"])
    target = data["churn"]
    _, X_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    predictions = model.predict(X_test)
    churn_index = list(model.classes_).index(1)
    churn_probabilities = model.predict_proba(X_test)[:, churn_index]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, churn_probabilities),
        "confusion": confusion_matrix(y_test, predictions),
    }


artifact_paths = (os.path.join("models", "scaler.pkl"), os.path.join("models", "model.pkl"))
artifact_signature = tuple(
    os.path.getmtime(path) if os.path.exists(path) else None for path in artifact_paths
)
scaler, model = load_artifacts(artifact_signature)
validation = get_validation_metrics(artifact_signature)

st.markdown('<div class="eyebrow">CUSTOMER RETENTION / DECISION SUPPORT</div>', unsafe_allow_html=True)
st.title("Bank customer churn")
st.markdown(
    '<p class="intro">Estimate a customer\'s likelihood of leaving based on their account profile. '
    'Use the score as a screening signal, not a guarantee.</p>',
    unsafe_allow_html=True,
)

if validation:
    st.subheader("Model validation")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Test accuracy", f"{validation['accuracy']:.1%}")
    metric_columns[1].metric("Churn recall", f"{validation['recall']:.1%}")
    metric_columns[2].metric("Churn precision", f"{validation['precision']:.1%}")
    metric_columns[3].metric("ROC-AUC", f"{validation['roc_auc']:.3f}")

    if validation["recall"] < 0.5:
        st.warning(
            "The model misses many customers who churn. Its accuracy is not the same as "
            "its ability to identify churners."
        )
    elif validation["precision"] < 0.5:
        st.warning(
            "This model catches more churners, but also flags many customers who stay. "
            "Treat alerts as a review list, not a final decision."
        )

    with st.expander("View test-set confusion matrix"):
        st.caption("Rows are actual outcomes; columns are predicted outcomes.")
        matrix = validation["confusion"]
        st.markdown(
            f"""
            <table class="confusion-matrix">
                <thead>
                    <tr><th>Actual outcome</th><th>Predicted stayed</th><th>Predicted churned</th></tr>
                </thead>
                <tbody>
                    <tr><th>Stayed</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr>
                    <tr><th>Churned</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr>
                </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.subheader("Customer profile")

with st.form("churn_prediction"):
    left_column, right_column = st.columns(2)

    with left_column:
        credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650)
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", min_value=18, max_value=100, value=38)
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)

    with right_column:
        balance = st.number_input("Account balance ($)", min_value=0.0, value=75000.0, step=1000.0)
        products_number = st.selectbox("Number of products", [1, 2, 3, 4])
        credit_card = st.selectbox("Has a credit card?", ["Yes", "No"])
        active_member = st.selectbox("Active member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated salary ($)", min_value=0.0, value=100000.0, step=1000.0)

    submitted = st.form_submit_button("Calculate churn risk", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "credit_score": credit_score,
                "gender": int(gender == "Male"),
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products_number,
                "credit_card": int(credit_card == "Yes"),
                "active_member": int(active_member == "Yes"),
                "estimated_salary": estimated_salary,
                "country_Germany": int(country == "Germany"),
                "country_Spain": int(country == "Spain"),
            }
        ]
    )

    numeric_columns = [
        "credit_score",
        "age",
        "tenure",
        "balance",
        "products_number",
        "estimated_salary",
    ]
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])
    input_data = input_data[list(model.feature_names_in_)]

    churn_index = list(model.classes_).index(1)
    churn_probability = model.predict_proba(input_data)[0][churn_index]
    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction")
    result_column, probability_column = st.columns([1, 1])

    with result_column:
        if prediction == 1:
            st.error("Higher churn risk: the model predicts this customer may leave.")
        else:
            st.success("Lower churn risk: the model predicts this customer will stay.")

    with probability_column:
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
        st.progress(churn_probability, text="Churn probability")

    st.caption("This estimate reflects the model's learned patterns and may be wrong for an individual customer.")