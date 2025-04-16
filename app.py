import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# === Load model and scaler ===
model = load_model("basketball_nn_model.h5")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# === Load team stats data ===
team_stats_df = pd.read_excel("data.xlsx")
team_stats_df['W-L'] = team_stats_df['W-L'].astype(str)

# Clean team names
team_stats_df['TeamName'] = team_stats_df['Team'].str.strip().str.replace('\xa0', ' ', regex=True)

# === Streamlit App UI ===
st.set_page_config(page_title="🏀 NCAA Win Predictor", layout="centered")
st.title("🏀 NCAA Basketball Game Predictor")

# Team dropdowns
team_stats_df['TeamName'] = team_stats_df['TeamName'].astype(str).str.strip()
team_names = sorted(team_stats_df['TeamName'].dropna().unique())


home_team = st.selectbox("Select Home Team", team_names, index=team_names.index("Duke") if "Duke" in team_names else 0)
away_team = st.selectbox("Select Away Team", team_names, index=team_names.index("Utah St.") if "Utah St." in team_names else 1)

# === Prediction Function ===
def build_game_row(home_team, away_team, team_stats_df, scaler, model):
    try:
        home = team_stats_df[team_stats_df['TeamName'].str.contains(home_team, case=False, na=False)].iloc[0]
        away = team_stats_df[team_stats_df['TeamName'].str.contains(away_team, case=False, na=False)].iloc[0]

        row = {
            'Home NetRtg': home['NetRtg'],
            'Away NetRtg': away['NetRtg'],
            'Home Ortg': home['ORtg'],
            'Away Ortg': away['ORtg'],
            'Home DRtg': home['DRtg'],
            'Away DRtg': away['DRtg'],
            'Home AdjT': home['AdjT'],
            'Away AdjT': away['AdjT'],
            'Home Luck': home['Luck'],
            'Away Luck': away['Luck']
        }

        input_df = pd.DataFrame([row])
        input_df = input_df[feature_columns]  # Ensure column order matches training
        input_scaled = scaler.transform(input_df)
        prob = model.predict(input_scaled)[0][0]

        return prob
    except Exception as e:
        st.error(f"Error building game row: {e}")
        return None

# === Prediction Display ===
if home_team and away_team:
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        if st.button("Predict Home Team Win Probability"):
            prob = build_game_row(home_team, away_team, team_stats_df, scaler, model)
            if prob is not None:
                st.success(f"🏠 {home_team} has a {prob * 100:.2f}% chance of winning against 🚌 {away_team}")
