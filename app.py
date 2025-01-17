import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import streamlit as st

# Streamlit app
def main():
    st.title('Customer Segmentation & Sales Forecasting Dashboard')

    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page:', ['Overview', 'Customer Segmentation', 'Sales Forecasting'])

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")

        # Adjust column names if changed
        expected_columns = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceDate', 'InvoiceNo']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Dataset is missing required columns: {', '.join(missing_columns)}")
            return

        # Preprocess dataset
        data.dropna(subset=['CustomerID'], inplace=True)
        data = data[data['Quantity'] > 0]
        data = data[data['UnitPrice'] > 0]
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

        # Handle each page
        if page == 'Overview':
            st.header('Dataset Overview')
            st.dataframe(data.head())

            # Sales trend
            sales_trend = data.groupby(data['InvoiceDate'].dt.date)['TotalPrice'].sum()
            st.subheader('Daily Sales Trend')
            st.line_chart(sales_trend)

        elif page == 'Customer Segmentation':
            st.header('Customer Segmentation')

            # Calculate RFM
            import datetime
            reference_date = data['InvoiceDate'].max() + datetime.timedelta(days=1)
            data['Recency'] = (reference_date - data['InvoiceDate']).dt.days
            rfm = data.groupby('CustomerID').agg({
                'Recency': 'min',
                'InvoiceNo': 'count',
                'TotalPrice': 'sum'
            }).rename(columns={'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

            # Normalize and cluster
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

            # Visualize
            st.write('Cluster Information')
            st.dataframe(rfm)

            # PCA visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(rfm_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=rfm['Cluster'], palette='viridis', ax=ax)
            ax.set_title('Customer Segmentation Clusters')
            st.pyplot(fig)

        elif page == 'Sales Forecasting':
            st.header('Sales Forecasting')

            # ARIMA
            sales_trend = data.groupby(data['InvoiceDate'].dt.date)['TotalPrice'].sum()
            model = ARIMA(sales_trend, order=(5, 1, 0))
            results = model.fit()

            # Forecast
            forecast = results.forecast(steps=30)
            fig, ax = plt.subplots(figsize=(10, 6))
            sales_trend.plot(ax=ax, label='Actual Sales')
            forecast.plot(ax=ax, label='Forecast', color='red')
            ax.set_title('ARIMA Sales Forecast')
            ax.legend()
            st.pyplot(fig)

            # Prophet
            prophet_data = sales_trend.reset_index()
            prophet_data.columns = ['ds', 'y']
            prophet = Prophet()
            prophet.fit(prophet_data)
            future = prophet.make_future_dataframe(periods=30)
            forecast = prophet.predict(future)

            # Prophet visualization
            fig = prophet.plot(forecast)
            st.pyplot(fig)
    else:
        st.warning("Please upload a dataset to proceed.")

if __name__ == '__main__':
    main()

