#  Light EDA Notebook for Airline No-Show Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/synthetic_flight_passenger_data.csv")

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
df.to_csv("../data/cleaned_flight_passenger_data.csv", index=False)
