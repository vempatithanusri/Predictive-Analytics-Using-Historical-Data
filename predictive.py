# Predictive Analytics Using Historical Data
# Sales Forecasting using Linear Regression

# Install Required Libraries
# pip install pandas numpy matplotlib scikit-learn streamlit plotly openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Streamlit Page Setup
st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")

# Title
st.title("📈 Predictive Analytics Dashboard")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Historical Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read Dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    # Check Required Columns
    required_columns = ['Date', 'Sales']

    if all(col in df.columns for col in required_columns):

        # Convert Date
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by Date
        df = df.sort_values('Date')

        # Convert Date to Numeric
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        # Features and Target
        X = df[['Days']]
        y = df['Sales']

        # Train Model
        model = LinearRegression()
        model.fit(X, y)

        # Predictions
        df['Predicted Sales'] = model.predict(X)

        # Future Prediction (Next 30 Days)
        future_days = np.arange(df['Days'].max() + 1,
                                df['Days'].max() + 31)

        future_dates = pd.date_range(
            start=df['Date'].max() + pd.Timedelta(days=1),
            periods=30
        )

        future_predictions = model.predict(future_days.reshape(-1, 1))

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast Sales': future_predictions
        })

        # Model Accuracy
        mae = mean_absolute_error(y, df['Predicted Sales'])
        r2 = r2_score(y, df['Predicted Sales'])

        # KPI Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))
        col2.metric("MAE", round(mae, 2))
        col3.metric("R² Score", round(r2, 2))

        # Historical vs Predicted
        st.subheader("📊 Historical vs Predicted Sales")

        fig1 = px.line(
            df,
            x='Date',
            y=['Sales', 'Predicted Sales'],
            title='Actual vs Predicted Sales'
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Future Forecast
        st.subheader("🔮 Future Sales Forecast")

        fig2 = px.line(
            future_df,
            x='Date',
            y='Forecast Sales',
            title='Next 30 Days Sales Forecast'
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Forecast Table
        st.subheader("📋 Forecast Data")
        st.dataframe(future_df)

        # Download Forecast
        csv = future_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Forecast Data",
            data=csv,
            file_name='sales_forecast.csv',
            mime='text/csv'
        )

    else:
        st.error("Dataset must contain Date and Sales columns.")

else:
    st.info("Please upload a CSV or Excel dataset.")
