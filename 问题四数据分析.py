import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Definition
years = np.array([2019, 2020, 2021, 2022, 2023])

# Data indicators
pet_count = np.array([99.8, 108.5, 115.4, 122.6, 130.2])  # Pet count (million)
pet_food_output_value = np.array([440.7, 727.3, 1554, 1508, 2793])  # Total pet food production value (CNY billion)
pet_food_export_value = np.array([154.1, 9.8, 12.2, 24.7, 39.6])  # Total pet food export value (USD billion)
market_size = np.array([33.2, 35.6, 38.9, 42.1, 45.5])  # Pet market size (USD billion)
pet_household_penetration = np.array([0.18, 0.2, 0.2, 0.2, 0.22])  # Pet household penetration rate
food_export_percentage = np.array([2.8745, 2.6897, 2.3228, 2.3002, 2.4563])  # Food export percentage of goods
import_volume_index = np.array([120.23, 126.07, 134.64, 139.98, 145.32])  # Import volume index
population_growth = np.array([0.3547, 0.2380, 0.0893, -0.0131, -0.1038])  # Population growth rate (annual %)
gdp_per_capita = np.array([10143.86, 10408.72, 12617.51, 12662.58, 12614.06])  # GDP per capita (USD)
global_market_size = np.array([1000, 1055, 1149.42, 1200, 1250])  # Global market size (USD billion)
exchange_rate_usd = np.array([6.9, 6.9, 6.45, 6.75, 7.12])  # Exchange rate (USD/CNY)
exchange_rate_eur = np.array([7.75, 8.02, 7.65, 7.5, 7.34])  # Exchange rate (EUR/CNY)
eu_pet_food_import = np.array([20.5, 21, 22, 24.4, 27.2])  # EU pet food import (EUR billion)
us_pet_food_import = np.array([15.3, 16, 17.5, 19, 21.3])  # US pet food import (USD billion)

# Create a DataFrame for easier manipulation
data = {
    'Pet Count': pet_count,
    'Output Value': pet_food_output_value,
    'Export Value': pet_food_export_value,
    'Market Size': market_size,
    'Household Penetration': pet_household_penetration,
    'Food Export %': food_export_percentage,
    'Import Volume Index': import_volume_index,
    'Population Growth': population_growth,
    'GDP per Capita': gdp_per_capita,
    'Global Market Size': global_market_size,
    'Exchange Rate USD': exchange_rate_usd,
    'Exchange Rate EUR': exchange_rate_eur,
    'EU Import': eu_pet_food_import,
    'US Import': us_pet_food_import
}
df = pd.DataFrame(data)

# 2. Correlation Analysis
corr_matrix = df.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# 3. Visualize Correlation Matrix with Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Chinese Pet Food Industry Factors')
plt.show()

# 4. Analysis and Summary
# Finding factors most correlated with Pet Food Output and Export Values
corr_with_output = corr_matrix['Output Value']
corr_with_export = corr_matrix['Export Value']

# Output results
print("\nFactors Correlated with Pet Food Output Value:")
print(corr_with_output.sort_values(ascending=False))

print("\nFactors Correlated with Pet Food Export Value:")
print(corr_with_export.sort_values(ascending=False))
