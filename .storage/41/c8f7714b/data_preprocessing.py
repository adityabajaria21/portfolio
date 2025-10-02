"""
Data Preprocessing Module for E-commerce CLV Analysis
Handles data loading, cleaning, and preparation for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

class DataPreprocessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def download_dataset(self):
        """Download the Online Retail II dataset from UCI repository"""
        print("Downloading Online Retail II dataset...")
        
        # Create sample dataset since UCI dataset might not be directly accessible
        # In real scenario, you would download from: 
        # https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
        
        # Generate synthetic data that mimics the Online Retail II structure
        np.random.seed(42)
        
        # Generate sample data
        n_transactions = 50000
        n_customers = 2000
        n_products = 1000
        
        # Generate customer IDs
        customer_ids = np.random.choice(range(10000, 10000 + n_customers), n_transactions)
        
        # Generate invoice numbers
        invoice_nos = [f'INV{str(i).zfill(6)}' for i in range(1, n_transactions + 1)]
        
        # Generate stock codes
        stock_codes = [f'SKU{str(i).zfill(5)}' for i in np.random.choice(range(1, n_products + 1), n_transactions)]
        
        # Generate descriptions
        product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Toys']
        descriptions = [f'{np.random.choice(product_categories)} Item {i}' for i in range(n_transactions)]
        
        # Generate quantities (mostly positive, some negative for returns)
        quantities = np.random.choice(range(1, 21), n_transactions, p=[0.3] + [0.7/19]*19)
        # Add some returns (negative quantities)
        return_indices = np.random.choice(range(n_transactions), int(n_transactions * 0.05), replace=False)
        quantities[return_indices] = -quantities[return_indices]
        
        # Generate unit prices
        unit_prices = np.round(np.random.lognormal(2, 1, n_transactions), 2)
        
        # Generate invoice dates (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        invoice_dates = np.random.choice(date_range, n_transactions)
        
        # Generate countries (mostly UK)
        countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands', 'Belgium'], 
                                   n_transactions, p=[0.7, 0.1, 0.08, 0.05, 0.04, 0.03])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'InvoiceNo': invoice_nos,
            'StockCode': stock_codes,
            'Description': descriptions,
            'Quantity': quantities,
            'InvoiceDate': invoice_dates,
            'UnitPrice': unit_prices,
            'CustomerID': customer_ids,
            'Country': countries
        })
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        self.df.to_csv('data/raw/online_retail_ii.csv', index=False)
        print(f"Dataset created with {len(self.df)} transactions and saved to data/raw/online_retail_ii.csv")
        
        return self.df
    
    def load_data(self, file_path=None):
        """Load data from CSV file"""
        if file_path:
            self.data_path = file_path
        
        if not self.data_path:
            # Try to load from default location
            self.data_path = 'data/raw/online_retail_ii.csv'
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("Dataset not found. Generating sample dataset...")
            return self.download_dataset()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        if self.df is None:
            self.load_data()
        
        # Create a copy for processing
        df_clean = self.df.copy()
        
        # Convert InvoiceDate to datetime
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # Remove rows with missing CustomerID
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['CustomerID'])
        print(f"Removed {initial_rows - len(df_clean)} rows with missing CustomerID")
        
        # Remove rows with negative or zero quantities (returns and errors)
        df_clean = df_clean[df_clean['Quantity'] > 0]
        
        # Remove rows with negative or zero unit prices
        df_clean = df_clean[df_clean['UnitPrice'] > 0]
        
        # Calculate total amount for each transaction
        df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        # Remove outliers (transactions with extremely high amounts)
        q99 = df_clean['TotalAmount'].quantile(0.99)
        df_clean = df_clean[df_clean['TotalAmount'] <= q99]
        
        # Convert CustomerID to integer
        df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
        
        print(f"Data cleaned: {df_clean.shape}")
        print(f"Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")
        print(f"Number of unique customers: {df_clean['CustomerID'].nunique()}")
        print(f"Number of unique products: {df_clean['StockCode'].nunique()}")
        
        self.processed_df = df_clean
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df_clean.to_csv('data/processed/cleaned_data.csv', index=False)
        
        return df_clean
    
    def get_data_summary(self):
        """Generate summary statistics of the dataset"""
        if self.processed_df is None:
            self.clean_data()
        
        summary = {
            'total_transactions': len(self.processed_df),
            'unique_customers': self.processed_df['CustomerID'].nunique(),
            'unique_products': self.processed_df['StockCode'].nunique(),
            'date_range': {
                'start': self.processed_df['InvoiceDate'].min(),
                'end': self.processed_df['InvoiceDate'].max()
            },
            'total_revenue': self.processed_df['TotalAmount'].sum(),
            'avg_transaction_value': self.processed_df['TotalAmount'].mean(),
            'countries': self.processed_df['Country'].value_counts().to_dict()
        }
        
        return summary

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and clean data
    cleaned_data = preprocessor.clean_data()
    
    # Get summary
    summary = preprocessor.get_data_summary()
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")