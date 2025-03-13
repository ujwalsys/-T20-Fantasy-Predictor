import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("fantasy_cricket_data.csv")

data = load_data()

# Streamlit UI
st.title("🏏 Fantasy Cricket AI Predictor – Gameathon Edition 🚀")
st.write("Create the best fantasy cricket team for Indian T20 League Match 30+")

# Display available players
st.subheader("📋 Available Players")
st.dataframe(data[['Player Name', 'Team', 'Runs Scored', 'Wickets Taken', 'Fantasy Points']])

# Prepare data for ML Model
features = data[['Runs Scored', 'Wickets Taken']]
labels = np.where(data['Fantasy Points'] > 100, 1, 0)  # 1 = Top Player, 0 = Average

# Train AI Model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.subheader(f"✅ AI Model Accuracy: {accuracy*100:.2f}%")

# User Input for Prediction
st.subheader("🔮 Predict Your Player's Performance")
selected_player = st.selectbox("Select a Player", data['Player Name'].unique())
player_stats = data[data['Player Name'] == selected_player]

# Default values from dataset
runs = st.number_input("Enter Expected Runs", min_value=0, max_value=200, value=int(player_stats['Runs Scored'].values[0]))
wickets = st.number_input("Enter Expected Wickets", min_value=0, max_value=10, value=int(player_stats['Wickets Taken'].values[0]))

# Predict Player Performance
if st.button("🔍 Predict Player Performance"):
    prediction = model.predict([[runs, wickets]])
    if prediction[0] == 1:
        st.success(f"✅ {selected_player} is a **Top Pick!**")
    else:
        st.warning(f"⚠️ {selected_player} might not be the best choice.")

# Team Optimization
st.subheader("🏆 Build Your Fantasy Team")
selected_players = st.multiselect("Select 11 Players", data['Player Name'].unique())

if st.button("⚡ Generate Best Fantasy Team"):
    team_points = data[data['Player Name'].isin(selected_players)]['Fantasy Points'].sum()
    st.write(f"🔥 Your team's total fantasy points: **{team_points}**")

    if team_points > 900:
        st.success("🚀 This is a Winning Team!")
    else:
        st.warning("⚠️ Try selecting better-performing players!")

st.sidebar.markdown("📌 **Tip:** Select players with high Fantasy Points & match impact.")
s