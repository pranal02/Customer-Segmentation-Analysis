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
from io import StringIO


# Apply dark theme
st.set_page_config(page_title="Customer Segmentation & Sales Forecasting", layout="wide")
dark_theme_css = """
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.css-1dp5vir, .css-1d391kg {
    background-color: #1c1e26 !important;
}
.css-1offfwp {
    background-color: #0e1117 !important;
}
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title('Customer Segmentation & Sales Forecasting Dashboard')

    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page:', ['Overview', 'Customer Segmentation', 'Sales Forecasting'])

    # Define the path to the local dataset
    dataset_path = 'data\OnlineRetail.csv'

    
    try:
        # Read the dataset from the local file with UTF-8 encoding
        st.info("Loading dataset from local file...")
        
        try:
            # Try loading with UTF-8 encoding
            data = pd.read_csv(dataset_path, encoding='utf-8')
            st.success("Dataset loaded successfully with UTF-8 encoding!")
            
        except UnicodeDecodeError:
            # If UTF-8 encoding fails, fallback to ISO-8859-1
            st.warning("UTF-8 encoding failed. Retrying with ISO-8859-1 encoding...")
            data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
            st.success("Dataset loaded successfully with ISO-8859-1 encoding!")

        # Display data preview
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Ensure that 'CustomerID' column exists before calling dropna
        if 'CustomerID' in data.columns:
            data.dropna(subset=['CustomerID'], inplace=True)
        else:
            st.warning("'CustomerID' column not found in the dataset.")
        
    except FileNotFoundError:
        st.error(f"Dataset not found at path: {dataset_path}")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please check the file format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

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

        # Additional charts
        st.subheader('Top 10 Products by Total Sales')
        top_products = data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)

        st.subheader('Sales by Country')
        sales_by_country = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sales_by_country.index, y=sales_by_country.values, palette='viridis')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.subheader('Sales Distribution by Invoice')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data['TotalPrice'], bins=30, kde=True, color='blue', ax=ax)
        st.pyplot(fig)

        st.subheader('Number of Transactions by Day of Week')
        data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
        transactions_by_day = data['DayOfWeek'].value_counts()
        st.bar_chart(transactions_by_day)

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

        # Additional charts
        st.subheader('Cluster Distribution')
        cluster_counts = rfm['Cluster'].value_counts()
        st.bar_chart(cluster_counts)

        st.subheader('Average Monetary Value by Cluster')
        cluster_monetary = rfm.groupby('Cluster')['Monetary'].mean()
        st.bar_chart(cluster_monetary)

        st.subheader('Average Frequency by Cluster')
        cluster_frequency = rfm.groupby('Cluster')['Frequency'].mean()
        st.bar_chart(cluster_frequency)

        st.subheader('Recency Distribution by Cluster')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Cluster', y='Recency', data=rfm.reset_index(), palette='viridis', ax=ax)
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

        # Additional charts
        st.subheader('Yearly Sales Distribution')
        yearly_sales = data.groupby(data['InvoiceDate'].dt.year)['TotalPrice'].sum()
        st.bar_chart(yearly_sales)

        st.subheader('Monthly Sales Distribution')
        monthly_sales = data.groupby(data['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
        st.line_chart(monthly_sales)

        st.subheader('Heatmap of Daily Sales')
        daily_sales_pivot = data.pivot_table(
            index=data['InvoiceDate'].dt.month,
            columns=data['InvoiceDate'].dt.day,
            values='TotalPrice',
            aggfunc='sum',
            fill_value=0
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(daily_sales_pivot, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader('Outlier Analysis for Sales')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data['TotalPrice'], color='red', ax=ax)
        st.pyplot(fig)


if __name__ == '__main__':
    main()

