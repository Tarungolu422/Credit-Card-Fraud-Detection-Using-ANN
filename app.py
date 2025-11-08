import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Explainable ANN: Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Explainable ANN: Credit Card Fraud Detection")
st.markdown("""
### 🧠 Project Goal
> Build a deep learning model (ANN) that can detect **fraudulent transactions** hidden inside **highly imbalanced financial data**,  
> while teaching **data preprocessing**, **imbalance handling**, and **evaluation beyond accuracy**.
---
""")

# --------------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------------
st.header("📥 Step 1: Load the Dataset")
uploaded = st.file_uploader("Upload your credit card dataset (must contain 'Class' column)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("Using built-in Kaggle dataset.")
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

st.write("First 5 Rows:")
st.dataframe(df.head())
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("""
💡 **Explanation:**  
Each row represents a credit card transaction.  
- Columns `V1 – V28` are anonymized features extracted via PCA.  
- `Amount` is the transaction amount.  
- `Class`: 0 = Legitimate, 1 = Fraudulent.
""")

# --------------------------------------------------------
# STEP 2: CHECK IMBALANCE
# --------------------------------------------------------
st.header("⚖️ Step 2: Understand Class Imbalance")

fraud_ratio = df["Class"].value_counts(normalize=True) * 100
fig, ax = plt.subplots()
sns.countplot(x="Class", data=df, ax=ax, palette="Set2")
st.pyplot(fig)
st.write(f"Fraudulent transactions: **{fraud_ratio[1]:.3f}%** of total")

st.markdown("""
💬 **Why this matters:**  
Accuracy can lie!  
If 99.8% of transactions are non-fraud, a model predicting *'No Fraud'* every time still gets **99.8% accuracy** 😲  
That’s why we’ll use **Precision, Recall, and AUC** to judge our model.
""")

# --------------------------------------------------------
# STEP 3: DATA PREPROCESSING
# --------------------------------------------------------
st.header("🧮 Step 3: Data Preprocessing")

st.markdown("""
We now:
1. Separate features (X) and target (y)  
2. Scale numeric features  
3. Split data (80% training, 20% testing)
""")

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
st.success("✅ Data scaled and split successfully!")

# --------------------------------------------------------
# STEP 4: BUILD ANN MODEL
# --------------------------------------------------------
st.header("🧠 Step 4: Build & Train the Artificial Neural Network")

st.markdown("""
**ANN Structure:**
- Input Layer → accepts all features (V1–V28, Amount, Time)
- Hidden Layers → learn patterns (non-linear relations)
- Output Layer → gives probability of fraud (Sigmoid function)

🧩 We'll use scikit-learn’s `MLPClassifier` (multi-layer perceptron).
""")

hidden1 = st.sidebar.slider("Hidden Layer 1 Neurons", 8, 64, 32, 4)
hidden2 = st.sidebar.slider("Hidden Layer 2 Neurons", 4, 32, 16, 2)
alpha = st.sidebar.slider("Regularization (alpha)", 0.0001, 0.01, 0.001)
max_iter = st.sidebar.slider("Training Iterations", 100, 500, 200, 50)

with st.spinner("Training ANN model..."):
    ann = MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2),
        activation='relu',
        solver='adam',
        alpha=alpha,
        max_iter=max_iter,
        random_state=42
    )
    ann.fit(X_train, y_train)
st.success("✅ ANN Model trained successfully!")

# --------------------------------------------------------
# STEP 5: MODEL EVALUATION
# --------------------------------------------------------
st.header("📊 Step 5: Evaluate the Model — Beyond Accuracy")

y_pred = ann.predict(X_test)
y_prob = ann.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax_roc.plot([0, 1], [0, 1], '--', color='gray')
ax_roc.set_title("ROC Curve")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

st.metric("Model AUC", f"{auc:.4f}")

st.markdown("""
💡 **Interpretation:**
- **Precision:** Out of all predicted frauds, how many were actually fraud.  
- **Recall:** Out of all actual frauds, how many were caught.  
- **AUC:** How well model separates fraud vs non-fraud (closer to 1 = better).

---
""")

# --------------------------------------------------------
# STEP 6: USER UNDERSTANDING SECTION
# --------------------------------------------------------
st.header("🧩 Step 6: How This ANN Works — Concept Recap")

st.markdown("""
| Concept | Explanation |
|----------|-------------|
| **Input Layer** | Takes all transaction features as numeric input. |
| **Hidden Layers** | Combine features with weights → learn hidden fraud patterns. |
| **Activation (ReLU)** | Introduces non-linearity so network can learn complex behaviors. |
| **Output Layer (Sigmoid)** | Produces probability between 0–1 → 1 means 'Fraud'. |
| **Training** | Adjusts weights to minimize prediction error using backpropagation. |
| **Evaluation** | Uses Precision, Recall, AUC instead of Accuracy (since data is imbalanced). |

💬 **Goal:** Detect rare fraudulent transactions without overpredicting false alarms.
""")

st.success("🎯 You’ve now built and understood an explainable ANN for Fraud Detection!")
st.markdown("👨‍💻 *Built by Tarun Kumar Rathore — Explainable Deep Learning Project (Python 3.13 Compatible)*")
