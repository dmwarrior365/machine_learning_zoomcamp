import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

def eval(model):
    # get the mean absolute error
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    oil_mae = mean_absolute_error(y_test["oil_rate"],y_pred[:,0])
    oil_r2 = r2_score(y_test["oil_rate"],y_pred[:,0])
    water_mae = mean_absolute_error(y_test["water_rate"],y_pred[:,1])
    water_r2 = r2_score(y_test["water_rate"],y_pred[:,1])
    print(f" Oil MAE :{oil_mae} , R2 :{oil_r2}")
    print(f" Water MAE :{water_mae} , R2 :{water_r2}")

df = pd.read_excel('/workspaces/machine_learning_zoomcamp/Midterm_Project/data/Volve production data.xlsx', sheet_name='Daily Production Data', parse_dates=True)
df_new = df[df["WELL_BORE_CODE"].isin(['NO 15/9-F-14 H', 'NO 15/9-F-15 D', 'NO 15/9-F-11 H', 'NO 15/9-F-12 H'])]
df_new.columns = df.columns.str.lower().str.replace(' ', '_')
df_new.reset_index(drop=True, inplace=True)

# Calculate oil and water rates
df_new["oil_rate"] = (df_new["bore_oil_vol"] / df_new["on_stream_hrs"])
df_new["water_rate"] = (df_new["bore_wat_vol"] / df_new["on_stream_hrs"])
df_new["oil_rate"] = df_new["oil_rate"].replace(np.inf, 0)
df_new["water_rate"] = df_new["water_rate"].replace(np.inf, 0)

df_new = df_new.drop(columns=['well_bore_code', 'npd_well_bore_code', 'npd_field_code', 'npd_field_name', 'npd_facility_code',
                              'npd_facility_name', 'bore_wi_vol', 'flow_kind', 'well_type', 'avg_choke_uom'])

# Select only numeric columns and apply forward fill
numeric_columns = df_new.select_dtypes(include=['float64', 'int64']).columns
df_new[numeric_columns] = df_new[numeric_columns].fillna(method='ffill')

# Remove remaining Null Values
df_new = df_new.dropna(axis=0).reset_index(drop=True)

df_final = df_new.drop(columns=['bore_oil_vol', 'bore_wat_vol', 'on_stream_hrs', 'avg_downhole_temperature', 'avg_dp_tubing', 'bore_gas_vol'])

features = ['avg_annulus_press', 'avg_choke_size_p', 'avg_wht_p', 'avg_downhole_pressure', 'avg_whp_p', 'dp_choke_size']
X = df_final[features]

# Targets for prediction
y = df_final[['oil_rate', 'water_rate']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Display the split data sizes
print(f'Training set size: {X_train.shape}, Testing set size: {X_test.shape}')

xgb = XGBRegressor(max_depth=10,n_estimators=1000,objective="reg:squarederror")
xgb.fit(X_train,y_train)

eval(xgb)

# Export Model
with open ('/workspaces/machine_learning_zoomcamp/Midterm_Project/model/xgb_model.pkl', 'wb') as file:
  pickle.dump(xgb, file)