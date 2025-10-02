# E-commerce Customer Segmentation and Lifetime Value (CLV) Prediction 🛍️

## Project Overview
This project analyzes customer behavior in an e-commerce setting to segment customers and predict their lifetime value using advanced analytics techniques.

## Business Problem
A company wants to understand its customer base better to create targeted marketing campaigns. The goal is to segment customers based on their purchasing behavior and predict their future lifetime value.

## Analytical Techniques Used
1. **RFM Analysis**: Segmenting customers based on Recency, Frequency, and Monetary value
2. **K-Means Clustering**: Unsupervised learning algorithm to group customers into distinct clusters
3. **Regression Models**: Gamma-Gamma and Beta-Geometric/Negative Binomial Distribution for CLV prediction

## Dataset
- **Source**: Online Retail II UCI Dataset
- **Description**: Transactional data for a UK-based online retail company
- **Features**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## Project Structure
```
ecommerce-clv-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_customer_segmentation.ipynb
│   └── 04_clv_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   └── clv_models.py
├── visualizations/
├── reports/
└── requirements.txt
```

## Key Insights Expected
- Customer segments based on purchasing behavior
- RFM scores and customer categorization
- Predicted Customer Lifetime Value
- Actionable recommendations for marketing campaigns

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run data preprocessing: `python src/data_preprocessing.py`
2. Perform RFM analysis: `python src/rfm_analysis.py`
3. Execute clustering: `python src/clustering.py`
4. Predict CLV: `python src/clv_models.py`

## Visualization Dashboard
The project includes interactive dashboards showing:
- Customer segments and their characteristics
- RFM distribution
- CLV predictions
- Marketing recommendations