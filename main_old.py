import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1. Load the dataset
data = pd.read_csv('housing.csv')

#2. Create income category for stratified sampling and a stratified test set
data['income_cat']= pd.cut(data['median_income'],
                           bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                           labels=[1, 2, 3, 4, 5])

split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set= data.loc[train_index].drop('income_cat', axis=1)
    strat_test_set= data.loc[test_index].drop('income_cat', axis=1)

# We will work with a copy of the training set
housing= strat_train_set.copy()

#3. Separate features and labels
housing_labels= housing['median_house_value'].copy()
housing= housing.drop('median_house_value', axis=1)

print(housing, housing_labels)

#4. List the numerical and categorical columns
num_attribs= housing.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs= housing.select_dtypes(include=[object]).columns.tolist()

#5. Create pipelines 
# For numerical data
num_pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# For categorical data
cat_pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Construct Full Pipeline
full_pipeline= ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

#6. Apply the pipeline to the data
housing_prepared= full_pipeline.fit_transform(housing)
print(housing_prepared)

#7. Train and evaluate models

# Linear Regression
lin_reg= LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred= lin_reg.predict(housing_prepared)
lin_rmse= root_mean_squared_error(housing_labels, lin_pred)
#print(f'Linear Regression RMSE: {lin_rmse}')
lin_rmses= -cross_val_score(lin_reg, housing_prepared, housing_labels,
                              scoring='neg_mean_squared_error', cv=10)
print(pd.Series(lin_rmses).describe())

# Decision Tree
dec_tree= DecisionTreeRegressor()
dec_tree.fit(housing_prepared, housing_labels)
dec_tree_pred= dec_tree.predict(housing_prepared)
#dec_tree_rmse= root_mean_squared_error(housing_labels, dec_tree_pred)
dec_rmses= -cross_val_score(dec_tree, housing_prepared, housing_labels,
                              scoring='neg_mean_squared_error', cv=10)
#print(f'Decision Tree RMSE: {dec_tree_rmse}')
print(pd.Series(dec_rmses).describe())

# Random Forest
rand_forest= RandomForestRegressor()
rand_forest.fit(housing_prepared, housing_labels)
rand_forest_pred= rand_forest.predict(housing_prepared)
rand_forest_rmse= root_mean_squared_error(housing_labels, rand_forest_pred)
#print(f'Random Forest RMSE: {rand_forest_rmse}') 
rand_rmses= -cross_val_score(rand_forest, housing_prepared, housing_labels,
                              scoring='neg_mean_squared_error', cv=10)
print(pd.Series(rand_rmses).describe())