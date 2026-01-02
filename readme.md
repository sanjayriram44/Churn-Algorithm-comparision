# 📉 Customer Churn Prediction — Decision-Aware Machine Learning

## 📌 Project Overview

This project implements an **end-to-end customer churn prediction system** using structured customer data.  
Rather than focusing only on model accuracy, the project emphasizes **decision-aware modeling**, showing how predicted probabilities are converted into actionable decisions through **threshold tuning and model comparison**.

The goal is to demonstrate how real-world machine learning systems are built, evaluated, and justified — not just how models are trained.

---

## 🎯 Problem Statement

Customer churn is costly for subscription-based businesses.  
The objective of this project is to:

- Predict the **probability that a customer will churn**
- Convert predictions into **business-relevant decisions**
- Compare multiple model families fairly
- Select a final model based on evidence, not complexity

---

## 🧠 Key Concepts Demonstrated

- Pipeline-safe feature engineering
- Class imbalance handling
- Probability-based evaluation
- ROC–AUC vs Precision–Recall tradeoffs
- Threshold tuning for decision-making
- Model comparison across different algorithms
- Interpretable conclusions

---

## 📂 Dataset

- **Telco Customer Churn Dataset** (Kaggle)
- Contains customer demographics, subscription details, and service usage
- Binary target variable: **Churn (Yes / No)**

---

## 🏗️ Project Structure

project/
│
├── notebooks/
│ ├── baseline.ipynb
│ ├── 04_threshold_analysis.ipynb
│ ├── 05_tree_models.ipynb
│ ├── 06_boosting.ipynb
│
├── src/
│ └── pipelines/
│ └── feature_engineering.py
│
├── artifacts/
│ ├── y_test.npy
│ ├── y_proba.npy
│ └── test_indices.npy
│
└── README.md

---

## 🔧 Feature Engineering

- Cleaned raw columns and handled missing values
- Encoded binary and categorical features
- Engineered **tenure buckets**
- Built a **custom sklearn-compatible transformer**
- Ensured no data leakage and full pipeline compatibility

All feature engineering is reusable, deterministic, and production-safe.

---

## 🤖 Modeling Approach

### 1️⃣ Baseline Model — Logistic Regression

- Implemented using a full sklearn pipeline
- Class imbalance handled via `class_weight="balanced"`
- Evaluation metric: **ROC–AUC ≈ 0.84**

This confirmed that the data contains strong predictive signal.

---

### 2️⃣ Threshold Analysis (Decision-Making Phase)

- Used predicted probabilities instead of hard labels
- Analyzed **precision–recall tradeoffs**
- Performed threshold sweeping from 0.1 → 0.9
- Selected a **high-recall operating point (~0.35)**

This step demonstrated how **one threshold choice dramatically changes outcomes**, even with the same model.

---

### 3️⃣ Model Comparison

Models evaluated using **ROC–AUC and PR–AUC**:

| Model | Key Observation |
|-----|----------------|
| Logistic Regression | Strong, well-calibrated baseline |
| Decision Tree | Underperformed due to high variance |
| Random Forest | Comparable to logistic regression |
| XGBoost / LightGBM | Modest improvements (~1–3% PR–AUC) |

**Key Insight:**  
Feature engineering and threshold tuning contributed more to performance than model complexity.

---

## 🏆 Final Model Choice

**Logistic Regression** was selected as the final model because it offers:

- Competitive performance
- Stable and interpretable probabilities
- Easy deployment
- Clear decision control via threshold tuning

---

## 📊 Key Results

- ROC–AUC ≈ **0.84**
- PR–AUC ≈ **0.61**
- High recall achievable with controlled precision
- Clear, explainable decision tradeoffs

---

## 🧩 What This Project Demonstrates

This project shows the ability to:

- Build ML pipelines correctly
- Evaluate models beyond accuracy
- Make threshold-based decisions
- Compare models fairly
- Communicate tradeoffs clearly
- Think like a real ML practitioner

---

## 🚀 Possible Extensions

- Model explainability (coefficients / SHAP)
- Hyperparameter tuning
- Deployment as a REST API
- Cost-sensitive optimization
- Real-time inference simulation

---

## 📝 Key Takeaway

> **In practical machine learning, decision-making strategy often matters more than model complexity.**

---

## 👤 Author

**Sanjay Sriram**  
Computer Engineering Undergraduate  
Machine Learning & AI Projects
