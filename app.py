import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def balance_classes(data: pd.DataFrame, random_state: int = 2) -> pd.DataFrame:
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=random_state)
    return pd.concat([legit_sample, fraud], axis=0).reset_index(drop=True)


def split_features_target(data: pd.DataFrame, target_col: str = "Class"):
    X = data.drop(columns=target_col)  # ✅ no axis=1 — columns= already implies it
    y = data[target_col]
    return X, y


def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 2):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return {"train_accuracy": train_acc, "test_accuracy": test_acc}


def predict_transaction(model, feature_values: np.ndarray) -> int:
    return int(model.predict(feature_values.reshape(1, -1))[0])


# ──────────────────────────────────────────────
# Input parsing
# ──────────────────────────────────────────────


def parse_input(raw_input: str) -> np.ndarray:
    values = [v.strip() for v in raw_input.split(",")]
    return np.array(values, dtype=np.float64)


# ──────────────────────────────────────────────
# Streamlit UI helpers
# ──────────────────────────────────────────────


def render_metrics(metrics: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{metrics['train_accuracy']:.2%}")
    col2.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")


def render_prediction(prediction: int) -> None:
    if prediction == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")


def render_input_section(expected_features: int):
    st.subheader("Transaction Feature Input")
    st.caption(f"Provide {expected_features} comma-separated numeric values.")
    raw_input = st.text_input("Feature values", placeholder="0.0, -1.23, 2.45, ...")
    submitted = st.button("Analyse Transaction", type="primary")
    return raw_input, submitted


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────


@st.cache_resource(show_spinner="Training model…")
def build_pipeline(filepath: str):
    data = load_data(filepath)
    data = balance_classes(data)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    return model, metrics, X_train.shape[1]


# ──────────────────────────────────────────────
# App entry point
# ──────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="centered")
    st.title("🔍 Credit Card Fraud Detection")
    st.write("Logistic Regression model trained on balanced transaction data.")

    try:
        model, metrics, n_features = build_pipeline("creditcard.csv")
    except FileNotFoundError:
        st.error(
            "Dataset `creditcard.csv` not found. Place it in the working directory."
        )
        st.stop()

    with st.expander("Model Performance", expanded=True):
        render_metrics(metrics)

    st.divider()

    raw_input, submitted = render_input_section(n_features)

    if submitted:
        if not raw_input.strip():
            st.warning("Please enter feature values before submitting.")
        else:
            try:
                features = parse_input(raw_input)
                if len(features) != n_features:
                    st.error(f"Expected {n_features} values, got {len(features)}.")
                else:
                    prediction = predict_transaction(model, features)
                    render_prediction(prediction)
            except ValueError:
                st.error(
                    "Invalid input — make sure all values are numeric and comma-separated."
                )


if __name__ == "__main__":
    main()
