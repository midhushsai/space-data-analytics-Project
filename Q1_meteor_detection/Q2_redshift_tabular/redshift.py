# ----------------------------------------
# Q2: Redshift Prediction (Tabular)
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("data/galaxy.csv")

# Check columns
print(data.columns)

# Assume 'redshift' is target
X = data.drop('redshift', axis=1)
y = data['redshift']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)
