import pandas as pd
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv('./po2_data.csv')

# Define your response variable
y = df['motor_updrs']

# Define your explanatory variables (features)
X = df.drop(['motor_updrs', 'total_updrs'], axis=1)

# Step 1: Select a significance level (e.g., 0.05)
alpha = 0.05

# Step 2: Fit your model with all features
X = sm.add_constant(X)  # Add a constant term (intercept)
model = sm.OLS(y, X).fit()

# Step 3 to Step 6: Implement backward elimination
while True:
    # Step 3: Identify the feature with the highest P-value
    max_p_value = model.pvalues.drop('const').max()
    
    # Step 4: If the highest P-value is greater than alpha, remove the feature
    if max_p_value > alpha:
        feature_to_remove = model.pvalues.idxmax()
        X = X.drop(feature_to_remove, axis=1)
        
        # Fit the model again with the updated features
        model = sm.OLS(y, X).fit()
    else:
        # Step 6: Stop when all remaining features have P-values <= alpha
        break

# Display the final selected features
selected_features = X.columns.drop('const')
print("Selected Features:")
print(selected_features)

# Create a new DataFrame with selected features
selected_df = df[['motor_updrs', 'total_updrs']].copy()
selected_df[selected_features] = df[selected_features]
selected_df.to_csv('./po2_backward_elimination_selected_features.csv', index=False)
