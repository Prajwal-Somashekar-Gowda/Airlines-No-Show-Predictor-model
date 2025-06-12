import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "no_show_model.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_flight_passenger_data.csv")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['Frequent_Flyer_Status'])

X = df[[
    "Seat_Class", "Frequent_Flyer_Status", "Check_in_Method", "Travel_Purpose",
    "Seat_Selected", "Gender", "Income_Level",
    "Bags_Checked", "Age", "Flight_Satisfaction_Score",
    "Delay_Minutes", "Booking_Days_In_Advance", "Price_USD"
]]
y = df["No_Show"]

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.fit(X, y)
joblib.dump(model, MODEL_PATH)
joblib.dump(preprocessor, ENCODER_PATH)
st.set_page_config(page_title="Airline No-Show Predictor", layout="centered")
st.title("✈️ Airline No-Show Predictor")

#Tabs for user input and batch prediction 
tabs = st.tabs(["🔮 Predict One Passenger", "Batch Prediction"])

with tabs[0]:
    st.markdown("Enter passenger details to predict the likelihood of a No-show.")

    with st.form("input_form"):
        passenger_id = st.text_input("Passenger ID", value="demo-passenger")
        airline = st.selectbox("Airline", df["Airline"].unique())
        seat_class = st.selectbox("Seat Class", df["Seat_Class"].unique())
        flyer_status = st.selectbox("Frequent Flyer Status", df["Frequent_Flyer_Status"].dropna().unique())
        checkin_method = st.selectbox("Check-in Method", df["Check_in_Method"].unique())
        travel_purpose = st.selectbox("Travel Purpose", df["Travel_Purpose"].unique())
        seat_selected = st.selectbox("Seat Selected", df["Seat_Selected"].unique())
        gender = st.selectbox("Gender", df["Gender"].unique())
        income = st.selectbox("Income Level", df["Income_Level"].unique())
        bags_checked = st.slider("Bags Checked", 0, 3, 1)
        age = st.slider("Age", 18, 85, 30)
        satisfaction = st.slider("Satisfaction Score", 0.0, 10.0, 7.5)
        delay = st.slider("Delay Minutes", 0, 300, 0)
        advance_days = st.slider("Days Before Booking", 0, 180, 30)
        price = st.slider("Price (USD)", 50.0, 1000.0, 300.0)

        submitted = st.form_submit_button("Predict No-Show")

        if submitted:
            input_df = pd.DataFrame({
                "Seat_Class": [seat_class],
                "Frequent_Flyer_Status": [flyer_status],
                "Check_in_Method": [checkin_method],
                "Travel_Purpose": [travel_purpose],
                "Seat_Selected": [seat_selected],
                "Gender": [gender],
                "Income_Level": [income],
                "Bags_Checked": [bags_checked],
                "Age": [age],
                "Flight_Satisfaction_Score": [satisfaction],
                "Delay_Minutes": [delay],
                "Booking_Days_In_Advance": [advance_days],
                "Price_USD": [price]
            })

            prediction = model.predict(input_df)[0]
            result_text = "Prediction: No-Show" if prediction else "Prediction: Will Show Up"

            st.success(f"Passenger ID: {passenger_id} | Airline: {airline} | {result_text}")
            st.subheader("Passenger Data:")
            input_df.insert(0, "Airline", airline)
            input_df.insert(0, "Passenger_ID", passenger_id)
            st.dataframe(input_df)

with tabs[1]:
    st.markdown("This section automatically predicts all passengers likely to No-Show.")

    X_all = X.copy()
    preds = model.predict(X_all)
    results = df.copy()
    results["Predicted_No_Show"] = preds
    no_shows = results[results["Predicted_No_Show"] == 1]

    st.write(f"Passengers predicted to No-Show: {len(no_shows)}")
    st.dataframe(no_shows.reset_index(drop=True))

st.caption("This app uses airline passenger data to predict flight no-shows.")