import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Read the dataset into a DataFrame
df = pd.read_csv("./po2_data.csv")

# Create a new DataFrame for the target variables ('motor_updrs' and 'total_updrs')
target_variables = df[['motor_updrs', 'total_updrs']]

# Separate explanatory variables (X) from response variables (y)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)

# Create a pipeline for data preprocessing
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('power_transform', PowerTransformer(method='yeo-johnson')),
])

# Fit and transform the explanatory variables (X)
X_transformed = preprocessor.fit_transform(X)

# Restore column names of explanatory variables
X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns)

# Concatenate the transformed X and target variables
transformed_df = pd.concat([X_transformed_df, target_variables], axis=1)

# Save the single dataset with transformed features and target variables
transformed_df.to_csv('po2_after_yeo-johnson.csv', index=False)
