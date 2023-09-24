import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('./po2_after_VIF.csv')

# Separate explanatory variables (X_motor) from the response variable (y_motor_updrs) for 'motor_updrs'
X_motor = df.drop(['motor_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']

# Separate explanatory variables (X_total) from the response variable (y_total_updrs) for 'total_updrs'
X_total = df.drop(['total_updrs'], axis=1)
y_total_updrs = df['total_updrs']

# Initialize your machine learning model (Decision Tree Regressor in this example)
model = DecisionTreeRegressor()

# Define the number of folds for cross-validation
k = 10

# Perform k-fold cross-validation for 'motor_updrs'
motor_updrs_scores = cross_val_score(model, X_motor, y_motor_updrs, cv=k, scoring='neg_mean_squared_error')
motor_updrs_rmse_scores = (-motor_updrs_scores)**0.5  # Calculate RMSE from negative MSE scores

# Perform k-fold cross-validation for 'total_updrs'
total_updrs_scores = cross_val_score(model, X_total, y_total_updrs, cv=k, scoring='neg_mean_squared_error')
total_updrs_rmse_scores = (-total_updrs_scores)**0.5  # Calculate RMSE from negative MSE scores

# Print the cross-validation RMSE scores for 'motor_updrs'
print("Cross-Validation RMSE Scores for 'motor_updrs':", motor_updrs_rmse_scores)
print("Mean RMSE for 'motor_updrs':", motor_updrs_rmse_scores.mean())

# Print the cross-validation RMSE scores for 'total_updrs'
print("Cross-Validation RMSE Scores for 'total_updrs':", total_updrs_rmse_scores)
print("Mean RMSE for 'total_updrs':", total_updrs_rmse_scores.mean())
