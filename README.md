# WiFi Network Traffic Analysis & Prediction

## Overview

This project analyzes WiFi network traffic data and uses machine learning techniques to predict network traffic based on historical data.

The project is developed using Python and provides a simple interactive interface for analyzing and predicting WiFi network traffic.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit
* Machine Learning

## Dataset

The project uses a CSV dataset containing WiFi network traffic information.

**Dataset:** `wifi_traffic.csv`

## Project Files

* `app.py` — Streamlit application used to run the project.
* `model.py` — Contains the machine learning model and prediction logic.
* `wifi_data.csv` — Dataset used for analysis and prediction.

## Project Workflow

1. Load the WiFi network traffic dataset.
2. Preprocess the data using Pandas.
3. Analyze the dataset and identify traffic patterns.
4. Train a machine learning model using Scikit-learn.
5. Generate predictions from the trained model.
6. Visualize the results using Matplotlib.
7. Display the analysis and predictions through Streamlit.

## How to Run

### 1. Install Required Libraries

```bash
pip install pandas scikit-learn matplotlib streamlit
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

## Purpose

The purpose of this project is to apply data analysis, visualization, and machine learning techniques to WiFi network traffic data and develop an interactive prediction-based application.

