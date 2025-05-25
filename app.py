import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Title and Description ---
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍷 Wine Quality Predictor using KNN")
st.markdown("""
This app predicts **wine quality** (Good ≥7 or Bad <7) based on user-input chemical properties.  
Built using the **CRISP-DM** framework and **K-Nearest Neighbors (KNN)** model.
""")

# --- Load Dataset ---
data = pd.read_csv("winequality-red.csv")
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)

# --- Preprocessing ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Train KNN Model ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Sidebar: User Input ---
st.sidebar.header("🍇 Input Wine Characteristics")
def user_input():
    return pd.DataFrame({
        col: [st.sidebar.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].median()))]
        for col in X.columns
    })

user_df = user_input()
user_df_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)

# --- Main Prediction Section ---
st.header("🔍 Predict Wine Quality")

if st.button("🚀 Predict Quality"):
    prediction = knn.predict(user_df_scaled)[0]
    probability = knn.predict_proba(user_df_scaled)[0][1]
    label = "Good Quality (≥7)" if prediction else "Bad Quality (<7)"
    color = "green" if prediction else "red"

    # Display Prediction Result
    st.markdown(f"""
    <div style='
        background-color: {color};
        padding: 16px;
        border-radius: 8px;
        color: white;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;'>
        Predicted Quality: {label}
    </div>
    """, unsafe_allow_html=True)

    # --- Probability Bar Chart ---
    st.subheader("📊 Probability of Good vs Bad Quality")
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
    sns.barplot(x=["Bad", "Good"], y=knn.predict_proba(user_df_scaled)[0], palette="rocket", ax=ax_bar)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Probability")
    for i, v in enumerate(knn.predict_proba(user_df_scaled)[0]):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig_bar)

    # --- Permutation Feature Importance ---
    st.subheader("📌 Top Influential Features (Permutation Importance)")
    result = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean
    }).sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="mako", ax=ax_imp)
    ax_imp.set_title("Feature Importance (Permutation Based)")
    st.pyplot(fig_imp)

# --- Footer ---
st.caption("Developed for Data Mining Final Project | Follows the CRISP-DM Framework")
