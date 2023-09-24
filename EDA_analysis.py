"""  Data Loading """
import numpy as np
import pandas as pd
df = pd.read_csv('./po2_data.csv')


# finding number of rows and columns
df.shape
#Show first 5 rows of the dataset 
df.head()
# Display basic information about the dataset
print(df.info())
df.columns

# Check for any missing values in the entire dataset
print(df.isnull().sum().sum())
# Check for duplicate rows in the dataset
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows:", duplicate_rows.shape[0])

####################################################################
####################################################################
""" Bar Diagram: total_updrs  """
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

ax = sns.countplot(x='sex', data=df, palette=["#1f77b4", "#ff7f0e"])
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Distribution of People by Sex ')
plt.xticks([0, 1], ['Male', 'Female'])

# Adding integer count numbers on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()
####################################################################

####################################################################
"""Central Tendencies """
# Calculate the mean, median (50%), and standard deviation
mean = df.mean()
median = df.median()
stdv = df.std()
# Create a new DataFrame to display the results
central_tendencies = pd.DataFrame({'Mean': mean, 'Median (50%)': median, 'Standard Deviation': stdv})
# Display the central tendencies DataFrame
print(central_tendencies)
####################################################################
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric columns for box plots, excluding the specified columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Flatten the axes array to loop through them
axes = axes.ravel()

# Plot histograms with KDE for the common features
for i, feature in enumerate(numeric_columns):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Remove any remaining empty subplots
for i in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
####################################################################

"""  Confidence Interval """
#pip install tabulate
import pandas as pd
import numpy as np
from scipy.stats import t
from tabulate import tabulate

# List of numeric features/columns in your dataset (excluding subject#, age, sex)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Initialize empty lists to store results
selected_features = []
means = []
ci_lower_values = []  # To store lower confidence interval values
ci_upper_values = []  # To store upper confidence interval values

# Loop through each numeric feature
for feature in numeric_columns:
    # Separate data for the feature
    data = df[feature]
    
    # Calculate mean and standard error
    mean = data.mean()
    std_error = data.std() / np.sqrt(len(data))
    
    # Calculate degrees of freedom for the t-distribution
    dof = len(data) - 1
    
    # Calculate the critical value for the t-distribution
    t_critical = t.ppf(0.975, dof)  # 95% confidence level
    
    # Calculate the margin of error
    margin_of_error = t_critical * (std_error)
    
    # Calculate the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    selected_features.append(feature)
    means.append(mean)
    ci_lower_values.append(ci_lower)
    ci_upper_values.append(ci_upper)

# Create a DataFrame to store the results
confidence_interval_results = pd.DataFrame({
    'Feature': selected_features,
    'Mean': means,
    'CI_Lower': ci_lower_values,
    'CI_Upper': ci_upper_values
})

# Sort the DataFrame in ascending order of mean values
sorted_results = confidence_interval_results.sort_values(by='Mean', ascending=False)

# Print the selected columns in a table format with column headers in a single line
print(tabulate(sorted_results[['Feature', 'Mean', 'CI_Lower', 'CI_Upper']], headers='keys', tablefmt='pretty', showindex=False))

####################################################################
####################################################################
""" Box Plotting with Outliers """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Select numeric columns for box plots, excluding the specified columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Specify the name of the column you want to appear first
desired_column = "column_to_appear_first"

# Reorder the 'numeric_columns' list to place the desired column at the beginning
if desired_column in numeric_columns:
    numeric_columns = [desired_column] + [col for col in numeric_columns if col != desired_column]

# Set up the Seaborn style
sns.set(style="whitegrid")

# Create a subplot grid
num_plots = len(numeric_columns)
num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
axes = axes.flatten()

# Initialize a list to store results
results = []

# Create individual box plots for each included column
for i, column in enumerate(numeric_columns):
    sns.boxplot(data=df, y=column, ax=axes[i], palette="Set3")
    
    axes[i].set_title(column, fontsize=15)
    axes[i].set_ylabel("Value", fontsize=12)
    
    # Calculate the IQR for the column
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify and print the outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    if not outliers.empty:
        min_outlier = outliers.min()
        max_outlier = outliers.max()
        results.append([column, df[column].mean(), df[column].median(), df[column].mode().iloc[0], min_outlier, max_outlier])
    else:
        results.append([column, df[column].mean(), df[column].median(), df[column].mode().iloc[0], "Not exist", "Not exist"])

# Remove any empty subplots
for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Box Plots for Selected Numeric Columns", fontsize=20)
plt.subplots_adjust(top=0.92)
plt.show()

# Print the results in tabular format
headers = ["Features", "Mean", "Median", "Mode", "Min_Outliers", "Max_Outliers"]
print(tabulate(results, headers, tablefmt="pipe"))
####################################################################
####################################################################


####################################################################
####################################################################
"""Scatter Plots"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset into a DataFrame
df = pd.read_csv("./po2_after_VIF.csv")
# Define the selected features you want to plot
selected_features = df

# Create a 5x5 grid of scatter plots with red trend lines and blue data points for 'motor_updrs'
plt.figure(figsize=(15, 15))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(5, 5, i)
    sns.regplot(x=feature, y='motor_updrs', data=df, scatter_kws={'alpha': 0.5, 'color': 'blue'}, line_kws={'color': 'red'})
    plt.title(f'{feature} vs. motor_updrs')

plt.tight_layout()
plt.show()


# Define the selected features you want to plot
selected_features = df

# Create a 5x5 grid of scatter plots with red trend lines and blue data points for 'total_updrs'
plt.figure(figsize=(15, 15))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(5, 5, i)
    sns.regplot(x=feature, y='total_updrs', data=df, scatter_kws={'alpha': 0.5, 'color': 'blue'}, line_kws={'color': 'red'})
    plt.title(f'{feature} vs. total_updrs')

plt.tight_layout()
plt.show()


####################################################################
####################################################################
"""Checking for the linearity"""

"""Using Rainbow Method"""
import statsmodels.stats.api as sms
import statsmodels.regression.linear_model as smf
import pandas as pd

df = pd.read_csv("./po2_after_VIF.csv")

# Function to perform Rainbow Test
def Rainbow_lin_test(Y, X):
    model = smf.OLS(Y, X).fit()
    lin_p = sms.linear_rainbow(model, frac=0.5)[1]
    result = "Fail"
    
    if lin_p > 0.1:
        result = "Pass"
    
    return pd.Series([lin_p, 0.1, result], index=['Rainbow linearity p value', 'Threshold', 'Result'])

# Separate explanatory variables (X) from response variables (y) for 'motor_updrs'
# X_motor_updrs = df.drop(['motor_updrs', 'total_updrs'], axis=1)
X_motor_updrs = df[['ppe']]
y_motor_updrs = df['motor_updrs']

# Separate explanatory variables (X) from response variables (y) for 'total_updrs'
X_total_updrs = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_total_updrs = df['total_updrs']

# Perform the Rainbow Test for 'motor_updrs'
result_motor_updrs = Rainbow_lin_test(y_motor_updrs, X_motor_updrs)
print("Rainbow Test for 'motor_updrs':")
print(result_motor_updrs)

# Perform the Rainbow Test for 'total_updrs'
result_total_updrs = Rainbow_lin_test(y_total_updrs, X_total_updrs)
print("Rainbow Test for 'total_updrs':")
print(result_total_updrs)
####################################################################
####################################################################

"""Checking for the Multi-collinearity"""
"""Heat Map"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset into a DataFrame
df = pd.read_csv("./po2_data.csv")
# Compute the correlation matrix
corr_matrix = df.corr()
# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Find pairs with correlation >= 0.95
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

# Print the high-correlation pairs
for pair in high_corr_pairs:
    print(f"High correlation ({corr_matrix.loc[pair[0], pair[1]]}): {pair[0]} and {pair[1]}")


####################################################################
####################################################################

"""Checking for the Multi-collinearity"""
"""VIF"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the original dataset
df = pd.read_csv('./po2_data.csv')

# Separate explanatory variables (X) from response variables (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)

# Calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [round(variance_inflation_factor(X.values, i), 2) for i in range(X.shape[1])]

# Print the VIF values
print("Variance Inflation Factor (VIF):")
print(vif)
####################################################################
####################################################################

"""Checking for the Multi-collinearity"""
"""Droping Higher VIF"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the original dataset
df = pd.read_csv('./po2_data.csv')

# Separate explanatory variables (X) from response variables (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)

# Define a list of variables with high VIF values to drop
variables_to_drop = ['jitter(rap)', 'jitter(ddp)', 'shimmer(apq3)', 'shimmer(dda)', 'shimmer(apq5)','shimmer(abs)','shimmer(%)','jitter(%)']

# Drop the specified variables from the DataFrame X
X_cleaned = X.drop(variables_to_drop, axis=1)

# Include 'motor_updrs' and 'total_updrs' in the cleaned DataFrame
X_cleaned['motor_updrs'] = df['motor_updrs']
X_cleaned['total_updrs'] = df['total_updrs']

# Calculate VIF for each remaining explanatory variable
vif_data_after_removal = pd.DataFrame()
vif_data_after_removal["Variable"] = X_cleaned.columns
vif_data_after_removal["VIF"] = [round(variance_inflation_factor(X_cleaned.values, i), 2) for i in range(X_cleaned.shape[1])]

# Print the updated VIF values
print("Variance Inflation Factor (VIF) after variable removal:")
print(vif_data_after_removal)

# Save the cleaned data with 'motor_updrs' and 'total_updrs' to a new CSV file
X_cleaned.to_csv("po2_after_VIF.csv", index=False)




####################################################################
####################################################################

"""Checking for the Multi-collinearity"""
"""PCA"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv("./po2_data.csv")

# Separate explanatory variables (X) from response variables (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)
y_motor_updrs = df['motor_updrs']
y_total_updrs = df['total_updrs']

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Define the number of principal components to keep (you can adjust this number)
n_components = 5  # Example: Keep the top 5 principal components

# Perform PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_standardized)

# Create a DataFrame with the PCA components and assign original column names
X_pca_df = pd.DataFrame(data=X_pca, columns=X.columns[:n_components])

# Concatenate PCA components with response variables
X_pca_df['motor_updrs'] = y_motor_updrs
X_pca_df['total_updrs'] = y_total_updrs

# Save the PCA-transformed data to a new CSV file
X_pca_df.to_csv("po2_after_PCA.csv", index=False)

X_pca_df.columns