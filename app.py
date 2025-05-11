import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(page_title="Breast Cancer AI Explorer", layout="wide")
st.title("🎗️ Leveraging AI in Breast Cancer Treatment: Interactive Explorer")

@st.cache_data
def load_data(uploaded_file=None):
    # Load and drop missing values
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("dataR2.csv")
    return df.dropna()

# Sidebar uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load data
df = load_data(uploaded_file)
st.sidebar.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns (after dropping missing values).")

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Classification", data=df, ax=ax)
st.pyplot(fig)

# Train SVM model
y = df["Classification"]
X = df.drop("Classification", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc = make_pipeline(StandardScaler(), SVC(C=3, gamma='scale', kernel='rbf'))
svc.fit(X_train, y_train)

# Prediction Interface
st.header("Make a Prediction with Trained SVM")
st.subheader("Input Features")
user_input = {}
for col in df.drop("Classification", axis=1).columns:
    val = st.number_input(f"{col}", value=float(df[col].median()))
    user_input[col] = val
if st.button("Predict Risk Category"):
    input_df = pd.DataFrame([user_input])
    pred = svc.predict(input_df)[0]
    st.write(f"**Predicted Classification:** {pred}")

# Save model
enable_save = st.button("Save trained SVM model")
if enable_save:
    joblib.dump(svc, "svm_model.pkl")
    st.success("Model saved as svm_model.pkl")
