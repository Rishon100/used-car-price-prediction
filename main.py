import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
df = pd.read_csv('data/vehicle.csv')

df = df[['Price','Year','Kilometer','Fuel Type','Transmission','Owner','Engine','Max Power']]
df['car_age'] = 2025 - df['Year']

df['Fuel Type'] = df['Fuel Type'].replace({
    'CNG + CNG': 'CNG',
    'Petrol + CNG': 'Hybrid',
    'Petrol + LPG': 'Hybrid'
})
df = df[df['Price'] < df['Price'].quantile(0.95)]
X = df.drop('Price', axis = 1)
y = df["Price"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model  = GradientBoostingRegressor(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 3,
    subsample = 0.8
    )
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Train:", train_score)
print("Test:", test_score)
print("difference: ",(train_score - test_score)*100)