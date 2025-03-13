import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("fantasy_cricket_data.csv")

data = load_data()

# Debugging: Print available column names
st.write("Columns in CSV:", data.columns.tolist())

# Streamlit UI
st.title("Fantasy Cricket AI Predictor 🏏")
st.write("Select your players and predict the best team!")

# Show available players
st.subheader("Available Players")
st.dataframe(data[['Player of the Match', 'Runs Scored', 'Wickets Taken', 'Fantasy Points']])

# Feature selection for ML model
features = data[['Runs Scored', 'Wickets Taken']]
labels = np.where(data['Fantasy Points'] > 100, 1, 0)  # 1 = Good player, 0 = Average player

# Train AI model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, labels)

# User input for prediction
st.subheader("Predict Your Team's Performance")
st.write("Select a player and predict their performance!")

selected_player = st.selectbox("Choose a Player", data['Player of the Match'].unique())
player_stats = data[data['Player of the Match'] == selected_player]

runs = st.number_input("Enter Expected Runs of Player", min_value=0, max_value=200, value=int(player_stats['Runs Scored'].values[0]))
wickets = st.number_input("Enter Expected Wickets of Player", min_value=0, max_value=10, value=int(player_stats['Wickets Taken'].values[0]))

# Make prediction
if st.button("Predict Player Performance"):
    prediction = model.predict([[runs, wickets]])
    if prediction[0] == 1:
        st.success(f"{selected_player} is a **Great Choice!** ✅")
    else:
        st.warning(f"{selected_player} might not be the best option. ❌")
