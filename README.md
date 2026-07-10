# California Housing Price Prediction 🏡

An end-to-end machine learning pipeline that predicts median house values across California using the classic California Housing dataset — from raw data to a reusable, saved model.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-39D353?style=for-the-badge)

---

## Project Overview

I built this as a **production-style ML pipeline**, not a one-off notebook — the goal was reproducibility. Every preprocessing step (imputation, scaling, encoding) is wrapped inside a single Scikit-Learn `Pipeline` + `ColumnTransformer`, so the exact same transformations used during training are guaranteed to run on new data at prediction time.

The final model is a **Random Forest Regressor**, achieving an **R² of 0.82** on the held-out test set, with an **18% reduction in MAE** through iterative feature engineering and hyperparameter tuning.

---

## Features

- ✅ **End-to-end preprocessing pipeline** — missing value imputation, feature scaling, one-hot encoding for categorical features, all wrapped in a single `ColumnTransformer` + `Pipeline`
- ✅ **Stratified train-test splitting** — ensures the test set is representative of the full income/housing distribution, not randomly skewed
- ✅ **Model training** — trained with `RandomForestRegressor`, benchmarked against Linear Regression and Decision Tree using k-fold cross-validation
- ✅ **Model persistence** — both the fitted pipeline and trained model are saved via Joblib (`pipeline.pkl`, `model.pkl`) for reuse without retraining
- ✅ **Automated CSV prediction** — drop in an `input.csv` with housing features, get back `output.csv` with predicted `median_house_value` — no manual preprocessing needed

---

## Project Structure

```
Califonia-housing-price-prediction/
├── main.py             — Current pipeline: preprocessing, training, prediction
├── main_old.py          — Earlier version of the pipeline (kept for reference)
├── requirements.txt      — Python dependencies
└── .gitignore
```

---

## How It Works

```
Raw Housing Data
     │
     ▼
Stratified Train-Test Split
     │
     ▼
ColumnTransformer
 ├── Numerical Pipeline   → SimpleImputer → StandardScaler
 └── Categorical Pipeline → OneHotEncoder
     │
     ▼
RandomForestRegressor (training)
     │
     ▼
Save Pipeline + Model (Joblib → pipeline.pkl, model.pkl)
     │
     ▼
input.csv → Trained Pipeline → output.csv (predicted median_house_value)
```

---

## Results

| Metric | Result |
|---|---|
| R² Score (test set) | **0.82** |
| MAE reduction | **18%** via feature engineering & hyperparameter tuning |
| Models benchmarked | Linear Regression, Decision Tree, Random Forest (k-fold cross-validation) |
| Records trained on | **20,000+** |
| EDA coverage | **10+ features** analyzed, 3 high-impact predictors identified |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| ML Framework | Scikit-Learn |
| Preprocessing | `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline` |
| Model | `RandomForestRegressor` |
| Splitting | `StratifiedShuffleSplit` |
| Persistence | Joblib |

---

## Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/webwizardaj/Califonia-housing-price-prediction.git
cd Califonia-housing-price-prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the pipeline
```bash
python main.py
```
Drop your housing data into `input.csv` in the expected format — the script handles preprocessing, prediction, and writes results to `output.csv` automatically.

---

## Key Learnings

- Building production-style, reproducible ML pipelines rather than notebook-only workflows
- Handling mixed numerical + categorical data cleanly with `ColumnTransformer`
- Feature engineering and stratified sampling to avoid biased splits
- Training, evaluating, and persisting models for real-world reuse
- Automating end-to-end prediction workflows from raw CSV to results

---

## Developer Details

| Field | Details |
|---|---|
| **Name** | Ashwani Jha |
| **Role** | Machine Learning \| AI/ML Engineer |
| **Project Type** | End-to-end personal project |
| **Tech Stack** | Python, Scikit-Learn, Pandas, NumPy, Joblib |
| **LinkedIn** | [linkedin.com/in/ashwani-jha-03ab14311](https://www.linkedin.com/in/ashwani-jha-03ab14311) |
| **Email** | [ashwanijha287@gmail.com](mailto:ashwanijha287@gmail.com) |
Real-world automation with CSV workflows
