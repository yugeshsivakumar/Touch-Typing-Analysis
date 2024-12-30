import streamlit as st
import pandas as pd
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt

# Load the trained model
with open('Model/wpm_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data for historical WPM and calculating the current average
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Feature engineering: Convert timestamp to numerical format (e.g., days since the first entry)
df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days

# Calculate the current average WPM
current_avg_wpm = df['wpm'].mean()

# Streamlit UI
st.title('WPM Prediction Model')

# Create sliders for user input
col1, col2 = st.columns([2, 3])
with col1:
    days = st.slider('Days', 0, 365, 0)
    months = st.slider('Months', 0, 12, 0)
    years = st.slider('Years', 0, 10, 0)

    # Display the current WPM
    st.write(f"Current average WPM: {current_avg_wpm:.2f}")

    # Total days for prediction
    total_days = days + (months * 30) + (years * 365)

    # Prediction logic
    if total_days == 0:
        predicted_wpm = current_avg_wpm
        future_date = pd.to_datetime("today")
    else:
        # Prepare input for prediction
        X_predict = pd.DataFrame({'days_since_start': [total_days]})
        predicted_change = model.predict(X_predict)[0]
        predicted_wpm = max(current_avg_wpm + predicted_change, 0)  # Avoid negative WPM
        future_date = pd.to_datetime("today") + timedelta(days=total_days)

    st.write(f"Predicted WPM for {future_date.strftime('%Y-%m-%d')}: {predicted_wpm:.2f}")

# Plot the predictions
with col2:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot historical data
    ax.plot(df['timestamp'], df['wpm'], label='Historical WPM', color='blue', marker='o')

    # Add prediction point
    if total_days > 0:
        ax.scatter(future_date, predicted_wpm, color='red', label=f'Predicted WPM', s=100)

    # Customize the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('WPM')
    ax.set_title('Historical WPM vs Predicted WPM')
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)