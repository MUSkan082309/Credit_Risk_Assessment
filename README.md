# 💳 Credit Risk Assessment using XGBoost

This machine learning project predicts the likelihood of a credit default using demographic and financial features of loan applicants. We train and evaluate a model using the XGBoost algorithm and optimize it through hyperparameter tuning.

---

## 📌 Problem Statement

Credit default prediction is essential for financial institutions to assess loan risks. This project builds a classifier to determine whether an applicant will default in the upcoming month based on their financial history and repayment behavior.

---

## 🧾 Dataset Overview

It includes:

- **Demographics**: Age, Gender, Education, Marriage
- **Credit Info**: Credit limit, past bill amounts, repayment status
- **Target**: `default.payment.next.month`  
  - 0 = No default
  - 1 = Default

---

## 🧹 Data Preprocessing

- Dropped `ID` column
- Mapped unknown categories in `EDUCATION` and `MARRIAGE` columns
- Renamed `PAY_0` to `PAY_1` for consistency
- Standardized features using `StandardScaler`

---

## ⚙️ Model

- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: `RandomizedSearchCV`
- **Best Parameters**:
  - `n_estimators`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.01

---

## 📈 Evaluation

- **10-Fold Cross-Validation Accuracy**:
  - Mean Score: **82.14%**
  - Standard Deviation: **±1.06%**

- **Boxplot of CV Accuracy Scores** is used to visualize model stability.

---

## 🧪 Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
