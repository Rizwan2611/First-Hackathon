import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
# Replace 'your_data.csv' with the path to your historical sales data
data = pd.read_csv('your_data.csv')

# Example of how your data might look:
# data = {
#     'product_name': ['Product A', 'Product B', ...],
#     'location': ['City 1', 'City 1', ...],
#     'previous_sales': [100, 150, ...],
#     'demand': [120, 130, ...]
# }

# Preprocessing: Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['product_name', 'location'])

# Define features (X) and target variable (y)
X = data.drop('demand', axis=1)  # Features
y = data['demand']                 # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Function to predict inventory for new data
def predict_inventory(new_data):
    new_data = pd.get_dummies(new_data, columns=['product_name', 'location'])
    # Ensure the new data has the same columns as the training data
    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    return model.predict(new_data)

# Example usage of predict_inventory function
# Replace this with the actual data you want to predict
new_data = pd.DataFrame({
    'previous_sales': [110],
    'product_name_Product A': [1],
    'product_name_Product B': [0],
    'location_City 1': [1],
    'location_City 2': [0]
})

predicted_demand = predict_inventory(new_data)
print(f'Predicted Inventory Level: {predicted_demand[0]}')
