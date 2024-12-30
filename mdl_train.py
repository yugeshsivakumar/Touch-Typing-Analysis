import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Feature engineering: Convert timestamp to numerical format (e.g., days since the first entry)
df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days

# Prepare the feature matrix (X) and target vector (y)
X = df[['days_since_start']]
y = df['wpm']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using XGBoost
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR2 Score: {r2:.2f}")

# Save the trained model
with open('Model/wpm_model_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)
