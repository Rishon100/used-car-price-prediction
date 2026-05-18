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

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("Cross-validation scores:", scores)
print("Average R2:", scores.mean())

print("\n--- Enter Car Details ---")

year = int(input("Enter Year: "))
km = int(input("Enter Kilometers Driven: "))
fuel = input("Enter Fuel Type (Petrol/Diesel/CNG/Hybrid): ")
trans = input("Enter Transmission (Manual/Automatic): ")
owner = input("Enter Owner (First/Second/Third): ")
engine = input("Enter Engine (e.g., 1197 cc): ")
power = input("Enter Max Power (e.g., 82 bhp): ")

# create dataframe
new_data = pd.DataFrame([{
    'Year': year,
    'Kilometer': km,
    'Fuel Type': fuel,
    'Transmission': trans,
    'Owner': owner,
    'Engine': engine,
    'Max Power': power
}])

# same feature engineering
new_data['car_age'] = 2025 - new_data['Year']

# same cleaning (IMPORTANT)
new_data['Fuel Type'] = new_data['Fuel Type'].replace({
    'CNG + CNG': 'CNG',
    'Petrol + CNG': 'Hybrid',
    'Petrol + LPG': 'Hybrid'
})

# encoding
new_data = pd.get_dummies(new_data)

# align columns with training data
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# prediction
prediction = model.predict(new_data)

print("\nPredicted Price:", int(prediction[0]))