🏡 California Housing Price Prediction

An end-to-end machine learning project that predicts median house values in California using a complete preprocessing and modeling pipeline.


---

🚀 Project Overview

This project builds a California Housing Price Prediction Model using the classic California Housing Dataset. It includes data preprocessing, feature engineering, model training, evaluation, and automated CSV-based predictions.

The model uses Random Forest Regression and a fully designed Scikit-Learn Pipeline to ensure reproducibility and clean workflow.


---

📁 Features

✔ End-to-End Machine Learning Pipeline

Handles missing values

Scales numerical features

One-hot encodes categorical features

Uses ColumnTransformer + Pipeline

Performs stratified train-test splitting


✔ Model Training

Trained using RandomForestRegressor

Saves both the preprocessing pipeline and model (pipeline.pkl, model.pkl)

Easily reusable for new predictions


✔ Automated CSV Prediction

Input: input.csv with housing features

Output: output.csv with predicted median_house_value

No manual processing required — the pipeline handles everything



---

🧰 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

StratifiedShuffleSplit

SimpleImputer, StandardScaler, OneHotEncoder

Pipeline, ColumnTransformer

RandomForestRegressor


Joblib for model persistence



---

📊 How It Works

1️⃣ Preprocessing Pipeline

The project builds separate pipelines for numerical and categorical attributes:

Numerical → Imputation + Scaling

Categorical → One-Hot Encoding


Combined using ColumnTransformer.

2️⃣ Model Training

Once the data is processed, a Random Forest Regressor is trained to predict the median house values.

3️⃣ Saving the Model

Both the preprocessing pipeline and trained model are saved using Joblib.

4️⃣ Making Predictions

Run the script to process any input.csv and generate output.csv automatically.

---

🎯 Key Learnings

Building production-style ML pipelines

Handling mixed (numerical + categorical) data

Feature engineering & stratified sampling

Training, tuning, and saving ML models

Real-world automation with CSV workflows
