import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Title and Description ---
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍷 Wine Quality Predictor using Logistic Regression")
st.markdown("""
This app predicts **wine quality** (Good ≥7 or Bad <7) based on user-input chemical properties. 
Built using the **CRISP-DM** framework and **Logistic Regression** model.
""")

# --- Load Dataset ---
data = pd.read_csv("winequality-red.csv")
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)

# --- Preprocessing ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Logistic Regression Model ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# --- Sidebar: User Input ---
st.sidebar.header("🍇 Input Wine Characteristics")
def user_input():
    return pd.DataFrame({col: [st.sidebar.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].median()))] for col in X.columns})

user_df = user_input()
user_df_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)

# --- Predict Button ---
st.header("🔍 Predict Wine Quality")
if st.button("🚀 Predict Quality"):
    prediction = logreg.predict(user_df_scaled)[0]
    probability = logreg.predict_proba(user_df_scaled)[0][1]
    label = "Good Quality (≥7)" if prediction else "Bad Quality (<7)"

    st.markdown(f"### 🏷️ **Predicted Quality:** {label}")
    st.progress(int(probability * 100))

    # --- Visual: Probability Bar Chart ---
    st.subheader("📊 Probability of Good Quality")
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Bad", "Good"], y=logreg.predict_proba(user_df_scaled)[0], palette="rocket", ax=ax_bar)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Probability")
    for i, v in enumerate(logreg.predict_proba(user_df_scaled)[0]):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig_bar)

    # --- SHAP Explanation ---
    st.subheader("🔍 SHAP Explanation")
    try:
        explainer = shap.Explainer(logreg, X_train)
        shap_values = explainer(user_df_scaled)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation error: {e}")

# --- Footer ---
st.caption("Developed for Data Mining Final Project | Follows the CRISP-DM Framework")