import pandas as pd
import statsmodels.api as sm
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read the dataset into a DataFrame
df = pd.read_csv('./po2_after_rescaling.csv')
df.columns

"""
BUILD AND EVALUATE LINEAR REGRESSION FOR 'motor_updrs'
"""

# Separate explanatory variables (X_motor) from the response variable (y_motor_updrs) for 'motor_updrs'
X_motor = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']

# Split the data into training and test sets for motor_updrs
X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X_motor, y_motor_updrs, test_size=0.2, random_state=42)

# Build and evaluate the linear regression model for 'motor_updrs'
X_train_motor = sm.add_constant(X_train_motor)
model_motor = sm.OLS(y_train_motor, X_train_motor).fit()
model_details_motor = model_motor.summary()
print("Model summary for Motor UPDRS:")
print(model_details_motor)

# Calculate R-squared for the Motor UPDRS model
r_squared_motor = model_motor.rsquared

"""
BUILD AND EVALUATE LINEAR REGRESSION FOR 'total_updrs'
"""

# Separate explanatory variables (X_total) from the response variable (y_total_updrs) for 'total_updrs'
X_total = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_total_updrs = df['total_updrs']

# Split the data into training and test sets for total_updrs
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_total, y_total_updrs, test_size=0.2, random_state=42)

# Build and evaluate the linear regression model for 'total_updrs'
X_train_total = sm.add_constant(X_train_total)
model_total = sm.OLS(y_train_total, X_train_total).fit()
model_details_total = model_total.summary()
print("\nModel summary for Total UPDRS:")
print(model_details_total)

# Calculate R-squared for the Total UPDRS model
r_squared_total = model_total.rsquared

####################################################################
####################################################################

"""
Baseline (Dummy) Model
"""

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

## Print R-squared values for both models
print("\nR-squared Values")
print("Motor UPDRS Model:")
print(f"- R-squared (R²): {r_squared_motor:.4f}\n")

print("Total UPDRS Model:")
print(f"- R-squared (R²): {r_squared_total:.4f}\n")

print("Baseline (Dummy) Model for Motor UPDRS:")
print(f"- R-squared (R²): {r2_base_motor:.4f}\n")

print("Baseline (Dummy) Model for Total UPDRS:")
print(f"- R-squared (R²): {r2_base_total:.4f}\n")
