import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import numpy as np

# Read the dataset into a DataFrame
df = pd.read_csv("./po2_data.csv")

# Separate explanatory variables (X) from response variables (y) for 'motor_updrs' and 'total_updrs'
X_motor_updrs = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']

X_total_updrs = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_total_updrs = df['total_updrs']

# Create a PowerTransformer
scaler = PowerTransformer()

# Apply the PowerTransformer to make all explanatory variables more Gaussian-looking for 'motor_updrs'
X_motor_updrs_transformed = scaler.fit_transform(X_motor_updrs)

# Restore column names of explanatory variables
X_motor_updrs_transformed_df = pd.DataFrame(X_motor_updrs_transformed, columns=X_motor_updrs.columns)

# Build and evaluate the linear regression model for 'motor_updrs' with transformed explanatory variables
X_motor_updrs_transformed_df = sm.add_constant(X_motor_updrs_transformed_df)
model_motor_updrs_transformed = sm.OLS(y_motor_updrs, X_motor_updrs_transformed_df).fit()
model_details_motor_updrs_transformed = model_motor_updrs_transformed.summary()
print("Model summary for Motor UPDRS with Transformed Explanatory Variables:")
print(model_details_motor_updrs_transformed)

# Apply the PowerTransformer to make all explanatory variables more Gaussian-looking for 'total_updrs'
X_total_updrs_transformed = scaler.fit_transform(X_total_updrs)

# Restore column names of explanatory variables
X_total_updrs_transformed_df = pd.DataFrame(X_total_updrs_transformed, columns=X_total_updrs.columns)

# Build and evaluate the linear regression model for 'total_updrs' with transformed explanatory variables
X_total_updrs_transformed_df = sm.add_constant(X_total_updrs_transformed_df)
model_total_updrs_transformed = sm.OLS(y_total_updrs, X_total_updrs_transformed_df).fit()
model_details_total_updrs_transformed = model_total_updrs_transformed.summary()
print("Model summary for Total UPDRS with Transformed Explanatory Variables:")
print(model_details_total_updrs_transformed)

# Split the data into training and test sets for motor_updrs
X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(X_motor_updrs_transformed_df, y_motor_updrs, test_size=0.2, random_state=42)

# Build and evaluate the linear regression model for 'motor_updrs'
X_train_motor = sm.add_constant(X_train_motor)
model_motor = sm.OLS(y_train_motor, X_train_motor).fit()
model_details_motor = model_motor.summary()
print("\nModel summary for Motor UPDRS:")
print(model_details_motor)

# Calculate R-squared for the Motor UPDRS model
r_squared_motor = model_motor.rsquared

# Split the data into training and test sets for total_updrs
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_total_updrs_transformed_df, y_total_updrs, test_size=0.2, random_state=42)

# Build and evaluate the linear regression model for 'total_updrs'
X_train_total = sm.add_constant(X_train_total)
model_total = sm.OLS(y_train_total, X_train_total).fit()
model_details_total = model_total.summary()
print("\nModel summary for Total UPDRS:")
print(model_details_total)

# Calculate R-squared for the Total UPDRS model
r_squared_total = model_total.rsquared

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
