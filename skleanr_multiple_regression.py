import pandas as pd
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame

df = pd.read_csv('./po2_data.csv')
# Separate explanatory variables (X) from the response variables (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']

# Split dataset into 60% training and 40% test sets
X_train, X_test, y_train_motor, y_test_motor, y_train_total, y_test_total = train_test_split(
    X, y_motor_updrs, y_total_updrs, test_size=0.4, random_state=0)

# Build a linear regression model for motor_updrs
model_motor = LinearRegression()
model_motor.fit(X_train, y_train_motor)

# Build a linear regression model for total_updrs
model_total = LinearRegression()
model_total.fit(X_train, y_train_total)

# Predict motor_updrs and total_updrs values on the test set
y_pred_motor = model_motor.predict(X_test)
y_pred_total = model_total.predict(X_test)

# Compute performance metrics for motor_updrs
mae_motor = metrics.mean_absolute_error(y_test_motor, y_pred_motor)
mse_motor = metrics.mean_squared_error(y_test_motor, y_pred_motor)
rmse_motor = math.sqrt(mse_motor)
r2_motor = metrics.r2_score(y_test_motor, y_pred_motor)

# Compute performance metrics for total_updrs
mae_total = metrics.mean_absolute_error(y_test_total, y_pred_total)
mse_total = metrics.mean_squared_error(y_test_total, y_pred_total)
rmse_total = math.sqrt(mse_total)
r2_total = metrics.r2_score(y_test_total, y_pred_total)

print("Motor UPDRS performance:")
print("MAE: ", mae_motor)
print("MSE: ", mse_motor)
print("RMSE: ", rmse_motor)
print("R^2: ", r2_motor)

print("\nTotal UPDRS performance:")
print("MAE: ", mae_total)
print("MSE: ", mse_total)
print("RMSE: ", rmse_total)
print("R^2: ", r2_total)

# Dummy (Baseline) Model
# Compute the mean of the training 'Motor UPDRS' and 'Total UPDRS' values
y_base_motor = np.mean(y_train_motor)
y_base_total = np.mean(y_train_total)

# Create arrays with the mean values replicated as predictions for the test set
y_pred_base_motor = np.full_like(y_test_motor, y_base_motor)
y_pred_base_total = np.full_like(y_test_total, y_base_total)

# Compute performance metrics for the baseline model for motor_updrs
mae_base_motor = metrics.mean_absolute_error(y_test_motor, y_pred_base_motor)
mse_base_motor = metrics.mean_squared_error(y_test_motor, y_pred_base_motor)
rmse_base_motor = math.sqrt(mse_base_motor)
r2_base_motor = metrics.r2_score(y_test_motor, y_pred_base_motor)

# Compute performance metrics for the baseline model for total_updrs
mae_base_total = metrics.mean_absolute_error(y_test_total, y_pred_base_total)
mse_base_total = metrics.mean_squared_error(y_test_total, y_pred_base_total)
rmse_base_total = math.sqrt(mse_base_total)
r2_base_total = metrics.r2_score(y_test_total, y_pred_base_total)

print("\nBaseline (Dummy) Model Performance for Motor UPDRS:")
print("MAE: ", mae_base_motor)
print("MSE: ", mse_base_motor)
print("RMSE: ", rmse_base_motor)
print("R^2: ", r2_base_motor)

print("\nBaseline (Dummy) Model Performance for Total UPDRS:")
print("MAE: ", mae_base_total)
print("MSE: ", mse_base_total)
print("RMSE: ", rmse_base_total)
print("R^2: ", r2_base_total)
