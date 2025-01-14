# Import Library
import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Model Development
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV
from hyperopt import hp, tpe, fmin, Trials

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Use for Hyperparameter Optimization
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# For exporting model
import joblib

import warnings
warnings.filterwarnings("ignore")

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {model.__class__.__name__}")
    print(f"MSE: {mse:.2f}, R2: {r2:.2f}\n")
    return model, y_pred

def analyze_overfitting_underfitting(cv_results):
    train_r2 = cv_results['train_r2']
    test_r2 = cv_results['test_r2']
    mean_train_r2 = np.mean(train_r2)
    mean_test_r2 = np.mean(test_r2)

    if mean_train_r2 - mean_test_r2 > 0.1:
        print("Potential Overfitting: Train R2 is significantly higher than Test R2.")
    elif mean_test_r2 < 0.5:
        print("Potential Underfitting: Test R2 is low.")
    else:
        print("Model seems well-balanced.")

def find_best_model(cross_val_results):
    best_model = None
    best_score = -np.inf

    for model_name, results in cross_val_results.items():
        test_r2_mean = np.mean(results['test_r2'])
        if test_r2_mean > best_score:
            best_score = test_r2_mean
            best_model = model_name

    print(f"Best Model: {best_model} with Mean Test R2: {best_score:.2f}")

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)
        self.params = kwargs  # Store parameters

    def set_params(self, **params):
        self.params.update(params)
        self.model.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.params

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def objective(params):
    dtrain = xgb.DMatrix(X_train, label=y_train)  # Convert data to DMatrix format
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda']
    }
    
    # Perform cross-validation
    cv_results = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=100,
        nfold=3,
        metrics='rmse',
        as_pandas=True,
        seed=42
    )
    
    # Return the mean RMSE from the cross-validation
    return {'loss': cv_results['test-rmse-mean'].min(), 'status': STATUS_OK}

df = pd.read_csv('/workspaces/machine_learning_zoomcamp/Capstone_Project/data/CO2_Emissions_Canada.csv')
df_clean = df[df['fuel_type'] != "N"]
df_clean['fuel_efficiency'] = (df_clean['fuel_consumption_city_(l/100_km)'] + df_clean['fuel_consumption_hwy_(l/100_km)']) / 2

categorical_columns = df_clean.select_dtypes(include=['object']).columns
# Apply one-hot encoding
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df_clean[column] = le.fit_transform(df_clean[column])
    label_encoders[column] = le

# Remove remaining Null Values
df_clean = df_clean.dropna(axis=0).reset_index(drop=True)

# Prepare data
X = df_clean.drop(columns=['fuel_consumption_city_(l/100_km)', 'co2_emissions(g/km)', 'fuel_consumption_hwy_(l/100_km)', 
                           'fuel_consumption_comb_(l/100_km)', 'fuel_consumption_comb_(mpg)'])
y = df_clean['co2_emissions(g/km)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display the split data sizes
print(f'Training set size: {X_train.shape}, Testing set size: {X_test.shape}')

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Lasso': Lasso(alpha=0.1),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVM': SVR(),
    'XGBoost': XGBRegressorWrapper(random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
}

trained_models = {}
predictions = {}
cross_val_results = {}
for name, model in models.items():
    print(f"Training {name}...")
    trained_model, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    trained_models[name] = trained_model
    predictions[name] = y_pred

    # Perform cross-validation
    print(f"Performing cross-validation for {name}...")
    cv_results = cross_validate(
        model, X_train, y_train, 
        scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'], 
        cv=5, return_train_score=True
    )
    cross_val_results[name] = cv_results
    print(f"Cross-validation results for {name}:")
    for metric, values in cv_results.items():
        if metric.startswith('test_') or metric.startswith('train_'):
            print(f"{metric}: {values}, Mean: {np.mean(values):.2f}, Std Dev: {np.std(values):.2f}")
    analyze_overfitting_underfitting(cv_results)
    print()

# Find the best model
find_best_model(cross_val_results)

# Hyperparameter for XGBoost
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
    'reg_lambda': hp.uniform('reg_lambda', 0, 10)
}

## Optimize the hyperparameters
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Number of iterations
    trials=trials
)

## Convert integer parameters to int for XGBoost compatibility
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

print("Best Hyperparameters:", best_params)

# Cross Validation Analyze
## Convert training data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)

## Define parameters based on tuned hyperparameters
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': best_params['n_estimators'],
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'reg_alpha': best_params['reg_alpha'],
    'reg_lambda': best_params['reg_lambda'],
    'seed': 42
}

## Perform cross-validation
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=best_params['n_estimators'],
    nfold=5, 
    metrics='rmse',
    as_pandas=True,
    seed=42
)

# Display cross-validation results
print(cv_results)

# Extract the mean RMSE from the cross-validation
mean_rmse = cv_results['test-rmse-mean'].min()
print(f"Mean RMSE from Cross-Validation: {mean_rmse:.2f}")

# Build the second model
xgb_model = xgb.XGBRegressor(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

## Evaluate
y_pred = xgb_model.predict(X_test)

## Calculate RMSE and R²
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Test Set RMSE: {rmse:.2f}")
print(f"Test Set R²: {r2:.2f}")

# Save the model
joblib.dump(xgb_model, "/workspaces/machine_learning_zoomcamp/Capstone_Project/model/xgboost_best_model.pkl")

# Feature Importance
# Get the feature names
feature_names = X.columns

X_train_df = pd.DataFrame(X_train, columns=feature_names)

# Extract feature importance from the trained model
feature_importance = xgb_model.feature_importances_

# Sort feature importance
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_feature_names = X_train_df.columns[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for XGBoost Model')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()