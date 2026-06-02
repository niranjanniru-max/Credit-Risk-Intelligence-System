# 🏦 Credit Risk Intelligence System

# Credit Risk Intelligence System

A machine learning app for credit risk prediction.

🚀 **Live Demo:** [Click here to view](https://credit-risk-intelligence-system-niranjanniru-max2jkwbf46jutu7y.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-86%25-brightgreen)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.84-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

> Predicts whether a loan applicant will default — built with a full production ML pipeline, deployed as an interactive Streamlit dashboard.

---

## 🎯 What It Does

A bank loan officer enters applicant details (income, credit score, loan amount, credit type etc.) and the system instantly returns:

- **Default probability** (0–100%)
- **Risk category** — Low / Medium / High
- **Top 5 factors** driving the prediction
- **Recommendation** — APPROVE / REVIEW / REJECT

---

## 📸 Demo

> ![image alt](https://github.com/niranjanniru-max/Credit-Risk-Intelligence-System/blob/48b03a7b9e95ddecc46beb32eac1324ed4e4d339/Screenshot%202026-05-31%20134508.png)
> ![image alt](https://github.com/niranjanniru-max/Credit-Risk-Intelligence-System/blob/48b03a7b9e95ddecc46beb32eac1324ed4e4d339/Screenshot%202026-05-31%20134552.png)

**High Risk applicant → 90.9% default probability 🔴**
**Low Risk applicant → 38.0% default probability 🟢**

---

## 📊 Dataset

- **Source:** [Loan Default Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- **Size:** 148,670 loan records
- **Target:** `Status` — 0 = Repaid, 1 = Defaulted
- **Class distribution:** 75% safe, 25% default (imbalanced)
- **Features:** 34 columns — loan amount, income, credit score, interest rate, property value, credit bureau type, and more

---

## 😤 The Real Story — What Actually Happened

This project took **far longer than expected** and hit every possible wall. Here's the honest journey:

---

### 🚨 Problem 1 — Fake 100% Accuracy (Data Leakage)

First run. Decision Tree → 100% accuracy. Random Forest → 100% accuracy.

Felt amazing. Then I questioned it.

After investigation, found `ID` and `year` columns were in the training data. The model was memorizing row IDs, not learning patterns. Dropped them. Accuracy dropped to honest range.

**Lesson learned:** 100% accuracy on real-world data is always a red flag.

---

### 🚨 Problem 2 — 99% Accuracy (Leakage via Missingness)

After fixing ID/year leakage, still getting 99%+ accuracy. Something was still wrong.

Ran a null pattern check:

```python
null_mask = data['rate_of_interest'].isnull()
print(data[null_mask]['Status'].value_counts())
print(data[~null_mask]['Status'].value_counts())
```

Output:
```
rate_of_interest IS null  → Status 1 = 36,439  (almost ALL defaulters)
rate_of_interest NOT null → Status 0 = 112,031 (ALL safe loans)
```

The bank only recorded interest rates for completed loans. Defaulters never got proper records. So null = defaulter was a perfect hidden signal. The model was cheating — not learning.

Dropped `rate_of_interest`, `Interest_rate_spread`, and `Upfront_charges` entirely. Accuracy dropped to 87%. That was the real number.

**Lesson learned:** Data leakage doesn't only come from wrong columns. It can come from NULL PATTERNS too. This is something even experienced data scientists miss.

---

### 🚨 Problem 3 — Predictions Completely Reversed

After fixing leakage, built Streamlit app and tested it.

High risk applicant (low income, bad credit score) → 2.5% default  
Low risk applicant (high income, great credit score) → 15% default

The model was predicting the opposite of reality.

Traced it back to the null filling strategy. Was filling `rate_of_interest` nulls with mean (~4.5%) before the leakage fix. This accidentally taught the model:
> "Mean interest rate = defaulter"

Because 36,000+ defaulters all got filled with the same mean value.

Fixed by dropping those columns entirely instead of filling them.

**Lesson learned:** How you handle missing values directly shapes what your model learns. Wrong imputation = wrong model behavior.

---

### 🚨 Problem 4 — EQUI Credit Bureau Dominance

After all fixes, model still behaving oddly for some inputs. Checked feature importances:

```
credit_type_EQUI    0.248  ← 25% of all decisions
income              0.168
Credit_Score        0.147
```

Investigated why EQUI dominates:

```python
data.groupby('credit_type')['Status'].mean()
```

Output:
```
EQUI    0.999935  ← 99.9% of EQUI customers defaulted
CRIF    0.162343
EXP     0.159854
CIB     0.158041
```

99.9% of all EQUI credit bureau customers in this dataset defaulted. Not a model bug — a real data pattern. Retained as a legitimate feature since credit bureau type is a genuine signal banks use.

**Lesson learned:** Always check dominant features with `groupby().mean()` against the target. One category can explain everything.

---

### ⏱️ Performance Pain

- First GridSearchCV run: **14+ minutes** for 24 candidates × 3 folds
- Switched to `HalvingGridSearchCV` — 3 rounds, eliminates weak models early, much faster
- IterativeImputer on 148,670 rows: slow but worth it for honest imputation
- Final pipeline with SMOTE + RF + GridSearch: solid but heavy

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn, imbalanced-learn |
| Data | pandas, numpy |
| Visualization | plotly, matplotlib |
| Deployment | Streamlit |
| Model Saving | joblib |
| IDE | VS Code |

---

## 🔁 ML Pipeline — Step by Step

```
Raw Data (148,670 rows, 34 columns)
        ↓
Drop leaking columns (ID, year, rate_of_interest, Interest_rate_spread, Upfront_charges)
        ↓
Separate numeric and categorical columns
        ↓
ColumnTransformer:
    Numeric  → IterativeImputer → StandardScaler
    Categorical → OneHotEncoder (handle_unknown='ignore')
        ↓
Train/Test Split (stratify=y, 75/25)
        ↓
imblearn Pipeline:
    Preprocessor → SMOTE → RandomForestClassifier
        ↓
GridSearchCV (ROC-AUC scoring, cv=3)
    max_depth: [10, 20, None]
    n_estimators: [100, 200]
    min_samples_split: [2, 5]
    max_features: [sqrt, log2]
        ↓
Best Model: RF (max_depth=20, n_estimators=200, min_samples_split=5, max_features=sqrt)
        ↓
joblib.dump() → Streamlit App
```

---

## 📈 Results

### Model Comparison (before GridSearch tuning)

| Model | Train Acc | Test Acc | ROC-AUC | Overfit |
|---|---|---|---|---|
| Logistic Regression | 0.76 | 0.75 | 0.71 | No ✅ |
| Decision Tree | 1.00 | 0.75 | 0.72 | Yes ⚠️ |
| Random Forest | 1.00 | 0.85 | 0.77 | Yes ⚠️ |
| KNN | 0.87 | 0.75 | 0.72 | Yes ⚠️ |

### Final Model (after GridSearch)

| Metric | Score |
|---|---|
| Test Accuracy | **86%** |
| ROC-AUC | **0.84** |
| Precision (default) | 0.77 |
| Recall (default) | 0.60 |
| F1 Score (default) | 0.68 |

### Why ROC-AUC matters more than accuracy here

Dataset is imbalanced — 75% safe, 25% default. A dumb model that predicts "safe" for everyone gets 75% accuracy. ROC-AUC measures how well the model **separates** the two classes regardless of imbalance.

---

## 🔍 Key Findings

| Finding | Detail |
|---|---|
| Data leakage via missingness | rate_of_interest null = 99.3% defaulters |
| EQUI bureau default rate | 99.9% of EQUI customers defaulted |
| Strongest real features | credit_type (19%), income (16%), loan_amount (14%), Credit_Score (14%) |
| SMOTE impact | Recall improved from 54% → 60% on default class |
| Best imputation | IterativeImputer outperformed mean/median filling |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/niranjanniru-max/Credit-Risk-Intelligence-System.git
cd Credit-Risk-Intelligence-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates .pkl files)
python Loan_Default_Project.py

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
Credit-Risk-Intelligence-System/
│
├── app.py                      # Streamlit dashboard
├── Loan_Default_Project.py     # Full training pipeline
├── credit_risk_model.pkl       # Saved trained model
├── model_columns.pkl           # Saved column list
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🧠 What I Learned

This was my **first real ML project** built from scratch as a 4th year B.Tech student with 2 months of ML knowledge. Here's what actually stuck:

**Technical:**
- Data leakage can be subtle — null patterns, not just wrong columns
- 100% accuracy is a warning sign, not a celebration
- `imblearn Pipeline` is required for SMOTE — regular sklearn Pipeline breaks
- `IterativeImputer` treats nulls as regression targets — smarter than mean
- `groupby().mean()` on target reveals which categories dominate
- `predict_proba()[:,1]` gives probability of the positive class
- GridSearchCV with `scoring='roc_auc'` is better than accuracy for imbalanced data
- Feature names after OneHotEncoding must be manually reconstructed for importance analysis

**Mindset:**
- Real data is messy and fights back
- Every "bug" was actually a learning opportunity
- Understanding WHY the model is wrong matters more than fixing it blindly
- 86% honest accuracy beats 99% fake accuracy every time

---

## 👤 Author

**Kamarouthu Niranjan Varma**
B.Tech Data Science — Dr. K.V. Subba Reddy Institute of Technology (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-niranjanniru--max-black)](https://github.com/niranjanniru-max)

---

## 📄 License

MIT License — free to use, modify, and distribute.
