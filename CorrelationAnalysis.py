# NMI和其他参数的相关性分析
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

save_name="results_overlap1"

# Load data
data = pd.read_csv(f'output_files/{save_name}_table.csv')

# Define independent variables (parameters) and dependent variable (NMI_score)
X = data[['adaptation_rate', 'map_dimensions', 'eps', 'min_samples', 'm']]
y = data['NMI_score']

# Standardize data for better interpretation
X_standardized = (X - X.mean()) / X.std()

# Add a constant for intercept in OLS model
X_standardized = sm.add_constant(X_standardized)

# Fit the OLS model
model = sm.OLS(y, X_standardized).fit()

# Print the summary of the model
summary = model.summary()
print(summary)

# Identify parameter with the highest effect based on p-values and coefficients
effect_params = pd.DataFrame({
    'Parameter': ['Intercept'] + X.columns.tolist(),
    'Coefficient': model.params,
    'P-value': model.pvalues
})

# Sort by absolute value of coefficient and highlight significant parameters (p < 0.05)
effect_params['Abs_Coefficient'] = np.abs(effect_params['Coefficient'])
effect_params = effect_params.sort_values(by='Abs_Coefficient', ascending=False)
effect_params['Significant'] = effect_params['P-value'] < 0.05

print("\nParameters sorted by their influence on NMI_score:\n")
print(effect_params)

# Save the detailed result to a CSV file
effect_params.to_csv(f'output_files/{save_name}_parameter_influence_analysis.csv', index=False)

print(f"\nAnalysis completed. Results saved to 'output_files/{save_name}_parameter_influence_analysis.csv'.")
