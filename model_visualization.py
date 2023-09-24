import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load each dataset
datasets = {
    'Original': pd.read_csv('po2_data.csv'),
    'Outliers Removed': pd.read_csv('cleaned_data.csv'),
    'PCA': pd.read_csv('po2_after_PCA.csv'),
    'VIF': pd.read_csv('po2_after_VIF.csv'),
    'Yeo-Johnson': pd.read_csv('po2_after_yeo-johnson.csv'),
    'Rescaling': pd.read_csv('po2_after_rescaling.csv'),
    'Backward Elimination': pd.read_csv('po2_backward_elimination_selected_features.csv')
}

# Initialize linear regression models for 'motor_updrs' and 'total_updrs'
motor_updrs_model = LinearRegression()
total_updrs_model = LinearRegression()

# Initialize lists to store R-squared scores for 'motor_updrs' and 'total_updrs'
r2_scores_motor_updrs = []
r2_scores_total_updrs = []
mae_motor_updrs = []
mae_total_updrs = []
mse_motor_updrs = []
mse_total_updrs = []
rmse_motor_updrs = []
rmse_total_updrs = []
dataset_names = list(datasets.keys())

# Calculate metrics for each dataset
for dataset_name, dataset in datasets.items():
    X = dataset.drop(['motor_updrs', 'total_updrs'], axis=1)
    y_motor_updrs = dataset['motor_updrs']
    y_total_updrs = dataset['total_updrs']
    
    # Fit the linear regression model for 'motor_updrs'
    motor_updrs_model.fit(X, y_motor_updrs)
    y_motor_updrs_pred = motor_updrs_model.predict(X)
    r2_motor_updrs = r2_score(y_motor_updrs, y_motor_updrs_pred)
    r2_scores_motor_updrs.append(r2_motor_updrs)
    mae_motor_updrs.append(mean_absolute_error(y_motor_updrs, y_motor_updrs_pred))
    mse_motor_updrs.append(mean_squared_error(y_motor_updrs, y_motor_updrs_pred))
    rmse_motor_updrs.append(np.sqrt(mean_squared_error(y_motor_updrs, y_motor_updrs_pred)))
    
    # Fit the linear regression model for 'total_updrs'
    total_updrs_model.fit(X, y_total_updrs)
    y_total_updrs_pred = total_updrs_model.predict(X)
    r2_total_updrs = r2_score(y_total_updrs, y_total_updrs_pred)
    r2_scores_total_updrs.append(r2_total_updrs)
    mae_total_updrs.append(mean_absolute_error(y_total_updrs, y_total_updrs_pred))
    mse_total_updrs.append(mean_squared_error(y_total_updrs, y_total_updrs_pred))
    rmse_total_updrs.append(np.sqrt(mean_squared_error(y_total_updrs, y_total_updrs_pred)))

# Create a 2x2 matrix of axes for each metric with numbers above the bars

metrics = {
    'R-squared': (r2_scores_motor_updrs, r2_scores_total_updrs),
    'MAE': (mae_motor_updrs, mae_total_updrs),
    'MSE': (mse_motor_updrs, mse_total_updrs),
    'RMSE': (rmse_motor_updrs, rmse_total_updrs)
}

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Linear Regression Model Comparison on Different Datasets', fontsize=16)

# Initialize handles and labels for the legend (only for the first plot)
handles = []
labels = []

for (metric_name, (motor_data, total_data)), ax in zip(metrics.items(), axs.flatten()):
    width = 0.35  # Width of each bar
    x = np.arange(len(dataset_names))  # X-axis values

    motor_bars = ax.bar(x - width/2, motor_data, width, alpha=0.8)
    total_bars = ax.bar(x + width/2, total_data, width, alpha=0.6)

    # Add numbers above the bars
    for i, j in zip(x, motor_data):
        ax.text(i - width/2, j + 0.01, f'{j:.4f}', ha='center', va='bottom')
    for i, j in zip(x, total_data):
        ax.text(i + width/2, j + 0.01, f'{j:.4f}', ha='center', va='bottom')

    ax.set_xlabel('Datasets')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45)
    
    # Collect handles and labels for the legend (only for the first plot)
    if ax == axs[0, 1]:
        handles.extend([motor_bars, total_bars])
        labels.extend(['motor_updrs', 'total_updrs'])

# Add a single legend for the first plot above the plot with proper colors
axs[0, 1].legend(handles, labels, loc='lower left', bbox_to_anchor=(0.5, .15), ncol=2)

# Remove legends for the other plots
for ax in axs[1:].flatten():
    ax.legend().set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
