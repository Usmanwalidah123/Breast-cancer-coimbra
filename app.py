import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Breast Cancer AI Explorer", layout="wide")
st.title("🎗️ Leveraging AI in Breast Cancer Treatment: Interactive Explorer")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("dataR2.csv")  # default path if no upload

@st.cache_resource
def train_models(df):
    y = df["Classification"]
    X = df.drop("Classification", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    # SVM on scaled data
    svc = make_pipeline(StandardScaler(), SVC(C=3, gamma='scale', kernel='rbf')).fit(X_train, y_train)
    # Random Forest
    rfc = RandomForestClassifier(n_estimators=90,
                                 max_depth=10,
                                 min_samples_split=10,
                                 min_samples_leaf=5,
                                 max_features='sqrt',
                                 random_state=52).fit(X_train, y_train)
    models = {"Logistic Regression": (lr, X_test, y_test),
              "SVM": (svc, X_test, y_test),
              "Random Forest": (rfc, X_test, y_test)}
    return models

# Sidebar - data upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load data
df = load_data(uploaded_file)
st.sidebar.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)

# EDA Section
st.header("Exploratory Data Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Missing Values")
    missing = df.isnull().sum()[df.isnull().sum() > 0]
    if not missing.empty:
        st.write(missing)
    else:
        st.write("No missing values.")
with col2:
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Classification", data=df, ax=ax)
    st.pyplot(fig)

# Statistics
st.subheader("Statistical Summary")
st.write(df.describe())

# Train models
st.header("Model Training & Evaluation")
models = train_models(df)

for name, (model, X_test, y_test) in models.items():
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, average='macro') * 100, 2)
    rec = round(recall_score(y_test, y_pred, average='macro') * 100, 2)
    st.subheader(f"{name}")
    st.write(f"Accuracy: {acc}%")
    st.write(f"Precision (macro): {prec}%")
    st.write(f"Recall (macro): {rec}%")
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Prediction Interface
st.header("Make a Prediction")
if st.checkbox("Use trained Random Forest model for custom input"):
    rfc = models["Random Forest"][0]
    st.subheader("Input Features")
    user_input = {}
    for col in df.drop("Classification", axis=1).columns:
        val = st.number_input(f"{col}", value=float(df[col].median()))
        user_input[col] = val
    if st.button("Predict Risk Category"):
        input_df = pd.DataFrame([user_input])
        pred = rfc.predict(input_df)[0]
        st.write(f"**Predicted Classification:** {pred}")

# Save model
if st.button("Download trained Random Forest model"):  
    joblib.dump(models["Random Forest"][0], "rfc_model.pkl")
    st.success("Model saved as rfc_model.pkl")
