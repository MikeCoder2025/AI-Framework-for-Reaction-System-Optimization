import time

start_time = time.time()


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skopt import gp_minimize
from skopt.space import Real
import numpy as np

# Load the initial dataset
file_path = 'datafile_updated.xlsx'  # Update this path if necessary
data = pd.ExcelFile(file_path)
dataset = data.parse('Sheet1')

# Define input and output columns
X = dataset[['Temperature', 'Massflow', 'Moleratio', 'Inletconc']]
y = dataset[['yCH4', 'yCO2', 'yO2']]

# Normalize the inputs and outputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Define the kernel for Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Initialize GP models for each output variable
gp_models = {}
for i, output in enumerate(['yCH4', 'yCO2', 'yO2']):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_scaled, y_scaled[:, i])
    gp_models[output] = gp

# Define the search space for Bayesian Optimization
search_space = [
    Real(250, 350, name='Temperature'),
    Real(20, 30, name='Massflow'),
    Real(2, 4, name='Moleratio'),
    Real(0.005, 0.025, name='Inletconc')
]

# Define the objective function
def objective_function(inputs):
    inputs_scaled = scaler_X.transform([inputs])
    yCH4_scaled = gp_models['yCH4'].predict(inputs_scaled)
    yCH4 = scaler_y.inverse_transform([[yCH4_scaled[0], 0, 0]])[0, 0]
    inlet_conc = inputs[3]
    conversion = (inlet_conc - yCH4) / inlet_conc
    if conversion < 0.5 or conversion > 1.0:
        return 1e6
    return -conversion

# Run Bayesian Optimization
result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=10,  # Limited calls for efficiency
    random_state=42
)

# Extract optimal conditions and maximum conversion
optimal_conditions = result.x
maximum_conversion = -result.fun

# Predict responses at optimal conditions
optimal_scaled = scaler_X.transform([optimal_conditions])
predicted_responses_scaled = [
    gp_models[output].predict(optimal_scaled)[0]
    for output in ['yCH4', 'yCO2', 'yO2']
]
predicted_responses = scaler_y.inverse_transform([predicted_responses_scaled])


# Display the results
print("\nOptimal Conditions:")
print(f"Temperature: {optimal_conditions[0]:.2f}")
print(f"Massflow: {optimal_conditions[1]:.2f}")
print(f"Moleratio: {optimal_conditions[2]:.2f}")
print(f"Inletconc: {optimal_conditions[3]:.4f}")
print(f"\nPredicted Responses:")
print(f"yCH4: {predicted_responses[0, 0]:.4f}")
print(f"yCO2: {predicted_responses[0, 1]:.4f}")
print(f"yO2: {predicted_responses[0, 2]:.4f}")
print(f"\nMaximum Conversion: {maximum_conversion:.4f}")


# ---- your existing code here ---- #

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")