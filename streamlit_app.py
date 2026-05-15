import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Logistic Classification", layout="wide")

st.title("Logistic Classification")


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    return df


def get_feature_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def preprocess_and_train(X: pd.DataFrame, y: pd.Series):
    # Lazy imports to keep app startup fast
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Identify numeric/categorical columns
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000)

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, acc


uploaded = st.sidebar.file_uploader(
    "Upload training CSV (includes target column)", type=["csv"], help="CSV containing features + a binary target column"
)

if uploaded is None:
    st.info(
        "Upload a CSV first. Then you can set feature values and click Predict."
    )
    st.stop()

# Always train automatically after upload (no separate Train button)
df = load_data(uploaded)
if df is None or df.empty:
    st.error("No data found in the uploaded file.")
    st.stop()

cols = df.columns.tolist()
target_col = st.sidebar.selectbox(
    "Select target column (binary 0/1)", options=cols
)

force_binary = st.sidebar.checkbox(
    "Force target to binary using 0/1 mapping (Non-zero -> 1)", value=False
)

# Build y/X from selected target column
if force_binary:
    y = (pd.to_numeric(df[target_col], errors="coerce").fillna(0) != 0).astype(int)
else:
    y = df[target_col]

X = df.drop(columns=[target_col]).copy()

with st.spinner("Training model from your CSV..."):
    model, acc = preprocess_and_train(X, y)

st.caption("Adjust inputs below and click Predict.")

st.subheader("Predict")

user_row = {}
for c in X.columns:
    if X[c].dtype.kind in "biufc":
        default = float(pd.to_numeric(X[c], errors="coerce").median()) if X[c].notna().any() else 0.0
        user_row[c] = st.number_input(f"{c}", value=float(default))
    else:
        mode = X[c].dropna().mode()
        default = str(mode.iloc[0]) if len(mode) else ""
        user_row[c] = st.text_input(f"{c}", value=default)

predict_btn = st.button("Predict")
result_placeholder = st.empty()

if predict_btn:
    X_in = pd.DataFrame([user_row])
    proba = model.predict_proba(X_in)[:, 1]
    pred = int(model.predict(X_in)[0])
    result_placeholder.success(
        f"Prediction: {pred}  |  Survival probability: {float(proba[0]):.4f}"
    )

st.sidebar.success(f"Trained. (test accuracy shown: {acc:.4f})")


