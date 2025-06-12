# ✈️ Airline No-Show Predictor

This project aims to predict whether a passenger will **not show up** for a scheduled flight using a real-world-style dataset. It simulates a machine learning workflow commonly used in the airline and travel industry to reduce revenue loss from unoccupied seats.

---

## Problem Statement

Airlines overbook flights to counter passenger no-shows. However, inaccurate forecasting can lead to empty seats or overbooking chaos. This tool helps identify passengers who are likely to not show up based on booking and travel details.

---

## Features
- **Single Passenger Prediction**: Enter travel details in a form and get instant no-show prediction
- **Batch Prediction Mode**: Automatically detects likely no-shows across the whole dataset
- **Model Persistence**: Uses joblib to save and reload model + preprocessor
- **Streamlit UI**: Interactive app for business/non-tech users

---

## Machine Learning Overview
- **Algorithm**: Random Forest Classifier
- **Pipeline**: Scikit-learn `Pipeline` with `ColumnTransformer` and `OneHotEncoder`
- **Features Used**:
  - Travel and personal details like: Seat Class, Flyer Status, Check-in Method, Travel Purpose
  - Booking behavior: Days in Advance, Delay Minutes
  - Demographics: Age, Gender, Income Level
- **Target**: `No_Show` (1 = no-show, 0 = attended)

---

## Exploratory Data Analysis (EDA)
- EDA included in `EDA.ipynb`
- Explores distribution of satisfaction scores, airline usage, and no-show correlations
- Feature importance from the Random Forest model
- Model performance metrics: Accuracy, Precision, Recall, F1 Score

---

## How to Run the App
### Step-by-step:
```bash
# 1. Navigate to root folder
cd No_Show_project

# 2. Install dependencies if needed
pip install -r requirements.txt  

# 3. Run the Streamlit app
streamlit run app/streamlit_app.py
```
---

## License
MIT License
