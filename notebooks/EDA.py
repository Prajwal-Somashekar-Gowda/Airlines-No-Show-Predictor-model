#  Light EDA Notebook for Airline No-Show Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset using path relative to this script so it works when run
# from any location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "synthetic_flight_passenger_data.csv")
df = pd.read_csv(data_path)

# Drop rows with missing Frequent Flyer Status
df.dropna(subset=["Frequent_Flyer_Status"], inplace=True)

# Summary
print("Total Records:", len(df))
print("No-Show Rate: {:.2f}%".format(df['No_Show'].mean() * 100))

# Target distribution
sns.countplot(data=df, x='No_Show')
plt.title("No-Show vs Show")
plt.xticks([0, 1], ['Show', 'No-Show'])
plt.ylabel("Count")
plt.show()

# No-Show by Seat Class
sns.countplot(data=df, x='Seat_Class', hue='No_Show')
plt.title("No-Show by Seat Class")
plt.xticks(rotation=45)
plt.show()

# No-Show by Frequent Flyer Status
sns.countplot(data=df, x='Frequent_Flyer_Status', hue='No_Show')
plt.title("No-Show by Frequent Flyer Tier")
plt.xticks(rotation=45)
plt.show()

# Satisfaction vs No-Show
sns.boxplot(data=df, x='No_Show', y='Flight_Satisfaction_Score')
plt.title("Satisfaction Score vs No-Show")
plt.xticks([0, 1], ['Show', 'No-Show'])
plt.show()

# Save a light-cleaned version
clean_path = os.path.join(script_dir, "..", "data", "cleaned_flight_passenger_data.csv")
df.to_csv(clean_path, index=False)
