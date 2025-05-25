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
st.title("\U0001F377 Wine Quality Classifier (CRISP-DM Framework)")
st.markdown("""
This web application uses machine learning to classify **wine quality** as either **Good (\u22657)** or **Bad (<7)**  
based on chemical properties. Built using the **CRISP-DM** data mining methodology.

\U0001F9D1‍\U0001F52C **Goal**: Help winemakers predict product quality based on lab measurements.  
\U0001F3AF **Success**: A model that accurately classifies wine quality and explains key factors.
""")

# --- Load and Display Data ---
st.header("\U0001F4C2 Dataset Overview")
data = pd.read_csv("winequality-red.csv")
st.write("**Source:** UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/ml/datasets/wine+quality))")
st.write("**Shape:**", data.shape)
st.write(data.head())

st.subheader("\U0001F4CA Data Summary")
st.write(data.describe())

st.subheader("\U0001F3AF Wine Quality Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.countplot(x='quality', data=data, palette='viridis', ax=ax1)
ax1.set_title("Wine Quality Value Counts")
ax1.set_xlabel("Wine Quality Score")
ax1.set_ylabel("Number of Samples")
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=10)
st.pyplot(fig1)

# --- Data Preparation ---
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Sidebar User Input ---
st.sidebar.header("\U0001F347 Input Wine Characteristics")
def user_input():
    return pd.DataFrame([{col: st.sidebar.number_input(col, value=float(data[col].median())) for col in X.columns}])
user_df = user_input()
user_df_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)

# --- OneR Algorithm ---
def one_r(X_train, y_train, X_test, y_test):
    best_acc, best_feature, best_rules = 0, None, {}
    for col in X_train.columns:
        train_round = X_train[col].round().astype(int)
        test_round = X_test[col].round().astype(int)
        rules = train_round.groupby(train_round).apply(lambda x: y_train[x.index].mode()[0])
        preds = test_round.map(rules).fillna(0).astype(int)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc, best_feature, best_rules = acc, col, rules
    final_preds = X_test[best_feature].round().astype(int).map(best_rules).fillna(0).astype(int)
    return final_preds, best_feature, best_rules

# --- Model Training ---
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0)
}

results = {}
oner_preds, best_feat, rules = one_r(X_train, y_train, X_test, y_test)
results["OneR"] = (accuracy_score(y_test, oner_preds), confusion_matrix(y_test, oner_preds), roc_auc_score(y_test, oner_preds))

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = (accuracy_score(y_test, preds), confusion_matrix(y_test, preds), roc_auc_score(y_test, preds))

# --- Evaluation ---
st.header("\U0001F4C8 Model Performance Evaluation")
st.dataframe(pd.DataFrame({
    "Accuracy": {k: v[0] for k, v in results.items()},
    "ROC AUC": {k: v[2] for k, v in results.items()},
    "Confusion Matrix": {k: str(v[1]) for k, v in results.items()}
}))

best_model = max(results.items(), key=lambda x: x[1][0])[0]
st.success(f"\U0001F3C6 Best Performing Model: {best_model} with {results[best_model][0]:.2%} accuracy.")

# --- ROC Curve ---
st.subheader("\U0001F50D ROC Curves")
fig2, ax2 = plt.subplots()
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax2.plot(fpr, tpr, label=name)
fpr, tpr, _ = roc_curve(y_test, oner_preds)
ax2.plot(fpr, tpr, label="OneR")
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
ax2.legend()
st.pyplot(fig2)

# --- SHAP Explanation ---
st.subheader("\U0001F4CC SHAP Explanation (Logistic Regression)")
try:
    explainer = shap.Explainer(models["Logistic Regression"], X_train)
    shap_values = explainer(user_df_scaled)
    st.write("**Prediction:**", int(models["Logistic Regression"].predict(user_df_scaled)[0]))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.warning(f"\u26A0\uFE0F SHAP explanation failed: {e}")

# --- Decision Tree Visualization ---
st.header("\U0001F333 Decision Tree Visualization")

# --- Feature Importance Bar Chart ---
st.subheader("📊 Top Features in the Decision Tree")
feature_importances = pd.Series(models["Decision Tree"].feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=True)  # ascending for horizontal layout

fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
sns.barplot(x=sorted_importances.values, y=sorted_importances.index, palette='crest', ax=ax_bar)

# Improve chart styling
ax_bar.set_title("Feature Importances from Decision Tree", fontsize=16, weight='bold')
ax_bar.set_xlabel("Importance Score", fontsize=12)
ax_bar.set_ylabel("Features", fontsize=12)
ax_bar.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(sorted_importances.values):
    ax_bar.text(v + 0.005, i, f"{v:.3f}", color='black', va='center', fontsize=10)

plt.tight_layout()
st.pyplot(fig_bar)

# --- Decision Tree Plot ---
st.subheader("🌳 Simplified Tree Diagram (Max Depth = 3)")
fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
plot_tree(
    models["Decision Tree"],
    feature_names=X.columns,
    class_names=["Bad", "Good"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False,
    proportion=True,
    ax=ax_tree,
    max_depth=3  # Limit depth for readability
)
fig_tree.suptitle("Decision Tree Classifier Structure", fontsize=18, weight='bold')
plt.tight_layout()
st.pyplot(fig_tree)


# --- Prediction Output ---
st.header("\U0001F4F2 Your Wine Prediction")
st.sidebar.markdown("### \U0001F916 Choose Model for Prediction")
selected_model = st.sidebar.selectbox("Model", list(models.keys()) + ["OneR"])

st.write("**Input Data:**")
st.dataframe(user_df)

if selected_model == "OneR":
    final_pred = "Good Quality" if rules.get(round(user_df[best_feat].values[0]), 0) else "Bad Quality"
    st.success(f"\u25B4 OneR (Best Feature: {best_feat}): {final_pred}")
else:
    prediction = models[selected_model].predict(user_df_scaled)[0]
    st.success(f"\U0001F50D {selected_model}: {'Good Quality' if prediction else 'Bad Quality'}")

st.caption("Made for Data Mining Final Project | CRISP-DM Framework | Streamlit App")
