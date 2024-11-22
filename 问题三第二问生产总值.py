import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
# 1. Data Definition
years = np.array([2019, 2020, 2021, 2022, 2023])  # Historical years
pet_count = np.array([99.8, 108.5, 115.4, 122.6, 130.2])  # Pet count (millions)
pet_food_output_value = np.array([440.7, 727.3, 1554, 1508, 2793])  # Total pet food output value in China (CNY billion)
pet_food_export_value = np.array([154.1, 9.8, 12.2, 24.7, 39.6])  # Total pet food export value in China (USD billion)
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # Pet market size (USD billion)
food_export_percentage = np.array([2.8745, 2.6897, 2.3228, 2.3002, 2.4563])  # Food export as percentage of commodity export
import_volume_index = np.array([120.23, 126.07, 134.64, 139.98, 145.32])  # Import volume index
population_growth = np.array([0.3547, 0.2380, 0.0893, -0.0131, -0.1038])  # Population growth rate (annual percentage)
gdp_per_capita = np.array([10143.86, 10408.72, 12617.51, 12662.58, 12614.06])  # GDP per capita (current USD)
global_market_size = np.array([1000, 1055, 1149.42, 1200, 1250])  # Global market size (USD billion)
exchange_rate = np.array([6.9, 6.7, 6.5, 6.4, 6.3])  # RMB to USD exchange rate
energy_price_index = np.array([90, 85, 88, 92, 95])  # Energy price index

# 2. Correlation Regression Model Construction
X_export = np.column_stack((food_export_percentage, population_growth, gdp_per_capita, exchange_rate, energy_price_index))  # Features related to export
X_import = np.column_stack((pet_count, market_size, import_volume_index, global_market_size, gdp_per_capita, energy_price_index))  # Features related to import

# Build regression models
export_model = LinearRegression().fit(X_export, pet_food_export_value)  # Export model
import_model = LinearRegression().fit(X_import, pet_food_output_value)  # Import model

# 3. Predict Population Growth Rate and GDP Per Capita for the Next Three Years
years_future = np.array([2025, 2026, 2027])  # Future three years
pop_growth_rate_future = np.array([-0.15, -0.18, -0.20])  # Future population growth rate
gdp_per_capita_future = gdp_per_capita[-1] * (1 + 0.05) ** np.arange(1, 4)  # GDP per capita growth rate 5%
exchange_rate_future = exchange_rate[-1] * (1 - 0.02) ** np.arange(1, 4)  # Assume exchange rate drops 2% per year
energy_price_index_future = energy_price_index[-1] * (1 + 0.03) ** np.arange(1, 4)  # Energy price index growth rate 3%

# 4. Build Pet Count Prediction Model for the Future
X_pet = np.column_stack((gdp_per_capita, population_growth))

# Linear Regression Model
pet_count_linear_model = LinearRegression().fit(X_pet, pet_count)
X_future_pet = np.column_stack((gdp_per_capita_future, pop_growth_rate_future))
pet_count_linear = pet_count_linear_model.predict(X_future_pet)

# Polynomial Regression Model
poly_order = 2
pet_count_poly_coeff = np.polyfit(years, pet_count, poly_order)
pet_count_poly = np.polyval(pet_count_poly_coeff, np.concatenate((years, years_future)))

# Nonlinear Regression Model (Exponential Model) - Using Scaled Years to Prevent Overflow
scaled_years = years - years.min()  # Scaling years to prevent overflow
scaled_future_years = np.concatenate((scaled_years, years_future - years.min()))

def exp_model(x, a, b):
    return a * np.exp(b * x)

try:
    popt, _ = curve_fit(exp_model, scaled_years, pet_count, maxfev=10000)
    pet_count_nonlinear = exp_model(scaled_future_years, *popt)
except RuntimeError as e:
    print(f"Nonlinear regression failed: {e}")
    pet_count_nonlinear = np.full(len(scaled_future_years), np.nan)

# 5. Weighted Combination to Predict Future Pet Count
# Ensure the predicted values have the same length
num_years = len(years)
regression_pred = pet_count_linear[:min(num_years, len(pet_count_linear))]
poly_pred = pet_count_poly[:min(num_years, len(pet_count_poly))]
nonlinear_pred = pet_count_nonlinear[:min(num_years, len(pet_count_nonlinear))]

# Pad the predictions to match lengths
max_length = max(len(regression_pred), len(poly_pred), len(nonlinear_pred))
regression_pred = np.pad(regression_pred, (0, max_length - len(regression_pred)), 'edge')
poly_pred = np.pad(poly_pred, (0, max_length - len(poly_pred)), 'edge')
nonlinear_pred = np.pad(nonlinear_pred, (0, max_length - len(nonlinear_pred)), 'edge')

# Initial weights
w0 = [1/3, 1/3, 1/3]

# Objective function: Minimize error (MSE)
def objective(w):
    return np.mean((w[0] * regression_pred + w[1] * poly_pred + w[2] * nonlinear_pred - pet_count[:max_length]) ** 2)

# Constraint: w1 + w2 + w3 = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bounds
bounds = [(0, 1), (0, 1), (0, 1)]

# Optimize weights using scipy.optimize.minimize
result = minimize(objective, w0, bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Predict future pet count using optimized weights
poly_future = pet_count_poly[-3:]
nonlinear_future = pet_count_nonlinear[-3:]
pet_count_weighted_future = (
    optimal_weights[0] * pet_count_linear +
    optimal_weights[1] * poly_future +
    optimal_weights[2] * nonlinear_future
)

# 6. Use Other Models to Predict Future Pet Food Export in China
# Support Vector Regression (SVR) Model
svr_export_model = SVR(kernel='rbf').fit(X_export, pet_food_export_value)
export_features_future = np.column_stack((food_export_percentage[-1] * (1 + 0.05) ** np.arange(1, 4), pop_growth_rate_future, gdp_per_capita_future, exchange_rate_future, energy_price_index_future))
pet_food_export_value_svr_future = svr_export_model.predict(export_features_future)

# Random Forest Regression Model
random_forest_export_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_export, pet_food_export_value)
pet_food_export_value_rf_future = random_forest_export_model.predict(export_features_future)

# 7. Visualization of Prediction Results
plt.figure(figsize=(10, 6))

# Historical pet food export value
plt.plot(years, pet_food_export_value, '-o', label='Historical Pet Food Export Value', linewidth=2, markersize=6, color='b')

# Support Vector Regression prediction
plt.plot(np.concatenate((years, years_future)), np.concatenate((pet_food_export_value, pet_food_export_value_svr_future)), '-s', label='Support Vector Regression', linewidth=2, markersize=6, color='r')

# Random Forest prediction
plt.plot(np.concatenate((years, years_future)), np.concatenate((pet_food_export_value, pet_food_export_value_rf_future)), '-^', label='Random Forest Regression', linewidth=2, markersize=6, color='g')

# Plot settings
plt.title('Prediction of Pet Food Export Value in China (2019-2027)')
plt.xlabel('Year')
plt.ylabel('Export Value (Billion USD)')
plt.legend()
plt.grid(True)
plt.show()

plt.suptitle('Forecast of China Pet Food Export Using Multiple Models', fontsize=16)
