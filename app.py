import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("wifi_data.csv")

# -----------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------
encoders = {}

for col in ["Day", "Time", "Location", "Device_Type", "Traffic"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = df.drop("Traffic", axis=1)
y = df["Traffic"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(
    page_title="College WiFi Dashboard",
    layout="wide"
)

st.title("📶 College WiFi Traffic Prediction Dashboard")

st.write(
    "This dashboard predicts WiFi traffic levels "
    "based on user activity and location data."
)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter Network Details")

day = st.sidebar.selectbox(
    "Day",
    encoders["Day"].classes_
)

time = st.sidebar.selectbox(
    "Time",
    encoders["Time"].classes_
)

location = st.sidebar.selectbox(
    "Location",
    encoders["Location"].classes_
)

device = st.sidebar.selectbox(
    "Device Type",
    encoders["Device_Type"].classes_
)

users = st.sidebar.slider(
    "Number of Users",
    0,
    200,
    50
)

duration = st.sidebar.slider(
    "Duration (Minutes)",
    0,
    120,
    30
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict Traffic"):

    input_data = [[
        encoders["Day"].transform([day])[0],
        encoders["Time"].transform([time])[0],
        encoders["Location"].transform([location])[0],
        users,
        encoders["Device_Type"].transform([device])[0],
        duration
    ]]

    prediction = model.predict(input_data)

    result = encoders["Traffic"].inverse_transform(prediction)

    st.success(f"Predicted Traffic Level: {result[0]}")

# -----------------------------
# ORIGINAL DATA FOR CHARTS
# -----------------------------
decoded_df = pd.read_csv("wifi_data.csv")

# -----------------------------
# CHART 1 - TRAFFIC DISTRIBUTION
# -----------------------------
st.subheader("Traffic Distribution")

traffic_counts = decoded_df["Traffic"].value_counts()

fig1, ax1 = plt.subplots()

traffic_counts.plot(
    kind="bar",
    ax=ax1
)

ax1.set_xlabel("Traffic Level")
ax1.set_ylabel("Count")
ax1.set_title("Traffic Distribution")

st.pyplot(fig1)

# -----------------------------
# CHART 2 - USERS BY LOCATION
# -----------------------------
st.subheader("Average Users by Location")

location_users = decoded_df.groupby(
    "Location"
)["Users"].mean()

fig2, ax2 = plt.subplots()

location_users.plot(
    kind="bar",
    ax=ax2
)

ax2.set_xlabel("Location")
ax2.set_ylabel("Average Users")
ax2.set_title("Average Users by Location")

st.pyplot(fig2)

# -----------------------------
# CHART 3 - TRAFFIC BY TIME
# -----------------------------
st.subheader("Traffic by Time")

time_counts = decoded_df["Time"].value_counts()

fig3, ax3 = plt.subplots()

ax3.pie(
    time_counts,
    labels=time_counts.index,
    autopct='%1.1f%%'
)

ax3.set_title("Usage by Time")

st.pyplot(fig3)

# -----------------------------
# CHART 4 - USERS TREND
# -----------------------------
st.subheader("Average Users Trend")

avg_users = decoded_df.groupby(
    "Time"
)["Users"].mean()

fig4, ax4 = plt.subplots()

avg_users.plot(
    kind="line",
    marker="o",
    ax=ax4
)

ax4.set_ylabel("Average Users")
ax4.set_title("Average Users Trend")

st.pyplot(fig4)

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(decoded_df)