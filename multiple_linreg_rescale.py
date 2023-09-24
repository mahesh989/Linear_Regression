import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the dataset into a DataFrame
df = pd.read_csv('./po2_data.csv')

# Separate explanatory variables (X) from response variables (y) for 'motor_updrs' and 'total_updrs'
X_updrs = df.drop(['motor_updrs', 'total_updrs'], axis=1)

# Initialize a StandardScaler
scaler = StandardScaler()

# Fit and transform the explanatory variables (features)
X_updrs_std = pd.DataFrame(scaler.fit_transform(X_updrs), columns=X_updrs.columns)

# Concatenate the rescaled explanatory variables with the original 'motor_updrs' and 'total_updrs' columns
df_std = pd.concat([df[['motor_updrs', 'total_updrs']], X_updrs_std], axis=1)

# Save the data with rescaled features and original response variables to a CSV file
df_std.to_csv('po2_after_rescaling.csv', index=False)

