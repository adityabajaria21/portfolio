"""
Customer Lifetime Value (CLV) Prediction Module
Implements Gamma-Gamma and BG/NBD models for CLV prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifetimes.utils import calibration_and_holdout_data
import warnings
warnings.filterwarnings('ignore')

class CLVPredictor:
    def __init__(self, data_path='data/processed/cleaned_data.csv'):
        self.data_path = data_path
        self.df = None
        self.summary_df = None
        self.bgf = None
        self.ggf = None
        self.clv_predictions = None
        
    def load_data(self):
        """Load cleaned transaction data"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            print(f"Transaction data loaded: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("Cleaned data not found. Please run data preprocessing first.")
            return None
    
    def prepare_clv_data(self):
        """Prepare data for CLV modeling"""
        if self.df is None:
            self.load_data()
        
        # Create summary data for CLV modeling
        # We need: frequency, recency, T (age), and monetary_value
        
        # Calculate observation period
        observation_period_end = self.df['InvoiceDate'].max()
        observation_period_start = self.df['InvoiceDate'].min()
        observation_period_days = (observation_period_end - observation_period_start).days
        
        print(f"Observation period: {observation_period_start} to {observation_period_end}")
        print(f"Total days: {observation_period_days}")
        
        # Group by customer and calculate required metrics
        customer_summary = self.df.groupby('CustomerID').agg({
            'InvoiceDate': ['min', 'max', 'count'],
            'TotalAmount': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        customer_summary.columns = ['CustomerID', 'first_purchase', 'last_purchase', 'frequency', 'total_amount', 'avg_amount']
        
        # Calculate recency (time between first and last purchase)
        customer_summary['recency'] = (customer_summary['last_purchase'] - customer_summary['first_purchase']).dt.days
        
        # Calculate T (time between first purchase and end of observation period)
        customer_summary['T'] = (observation_period_end - customer_summary['first_purchase']).dt.days
        
        # Frequency should be number of repeat purchases (total purchases - 1)
        customer_summary['frequency'] = customer_summary['frequency'] - 1
        
        # Monetary value should be average purchase amount (excluding first purchase for BG/NBD model)
        # For customers with only one purchase, we'll use their purchase amount
        customer_summary['monetary_value'] = customer_summary['avg_amount']
        
        # Remove customers with negative recency or T
        customer_summary = customer_summary[
            (customer_summary['recency'] >= 0) & 
            (customer_summary['T'] > 0) &
            (customer_summary['monetary_value'] > 0)
        ]
        
        # For BG/NBD model, we need customers with at least one repeat purchase for monetary modeling
        print(f"Customers prepared for CLV modeling: {len(customer_summary)}")
        print(f"Customers with repeat purchases: {len(customer_summary[customer_summary['frequency'] > 0])}")
        
        self.summary_df = customer_summary[['CustomerID', 'frequency', 'recency', 'T', 'monetary_value']].copy()
        
        # Save summary data
        self.summary_df.to_csv('data/processed/clv_summary_data.csv', index=False)
        
        return self.summary_df
    
    def fit_bgf_model(self):
        """Fit Beta-Geometric/NBD model for purchase frequency prediction"""
        if self.summary_df is None:
            self.prepare_clv_data()
        
        print("Fitting BG/NBD model...")
        
        # Initialize and fit the BG/NBD model
        self.bgf = BetaGeoFitter(penalizer_coef=0.01)
        self.bgf.fit(
            frequency=self.summary_df['frequency'],
            recency=self.summary_df['recency'],
            T=self.summary_df['T']
        )
        
        print("BG/NBD Model Summary:")
        print(self.bgf.summary)
        
        return self.bgf
    
    def fit_gamma_gamma_model(self):
        """Fit Gamma-Gamma model for monetary value prediction"""
        if self.summary_df is None:
            self.prepare_clv_data()
        
        if self.bgf is None:
            self.fit_bgf_model()
        
        print("Fitting Gamma-Gamma model...")
        
        # Filter customers with repeat purchases for Gamma-Gamma model
        repeat_customers = self.summary_df[self.summary_df['frequency'] > 0]
        
        # Initialize and fit the Gamma-Gamma model
        self.ggf = GammaGammaFitter(penalizer_coef=0.01)
        self.ggf.fit(
            frequency=repeat_customers['frequency'],
            monetary_value=repeat_customers['monetary_value']
        )
        
        print("Gamma-Gamma Model Summary:")
        print(self.ggf.summary)
        
        return self.ggf
    
    def predict_clv(self, prediction_period_days=365):
        """Predict Customer Lifetime Value"""
        if self.bgf is None:
            self.fit_bgf_model()
        
        if self.ggf is None:
            self.fit_gamma_gamma_model()
        
        print(f"Predicting CLV for {prediction_period_days} days...")
        
        # Predict number of purchases in the next period
        predicted_purchases = self.bgf.predict(
            t=prediction_period_days,
            frequency=self.summary_df['frequency'],
            recency=self.summary_df['recency'],
            T=self.summary_df['T']
        )
        
        # Predict average transaction value
        # For customers with no repeat purchases, use their historical average
        predicted_avg_value = np.zeros(len(self.summary_df))
        
        # For repeat customers, use Gamma-Gamma model
        repeat_mask = self.summary_df['frequency'] > 0
        if repeat_mask.sum() > 0:
            predicted_avg_value[repeat_mask] = self.ggf.conditional_expected_average_profit(
                frequency=self.summary_df.loc[repeat_mask, 'frequency'],
                monetary_value=self.summary_df.loc[repeat_mask, 'monetary_value']
            )
        
        # For single-purchase customers, use their historical value
        single_purchase_mask = self.summary_df['frequency'] == 0
        predicted_avg_value[single_purchase_mask] = self.summary_df.loc[single_purchase_mask, 'monetary_value']
        
        # Calculate CLV
        predicted_clv = predicted_purchases * predicted_avg_value
        
        # Calculate probability of being alive
        prob_alive = self.bgf.conditional_probability_alive(
            frequency=self.summary_df['frequency'],
            recency=self.summary_df['recency'],
            T=self.summary_df['T']
        )
        
        # Create results dataframe
        clv_results = self.summary_df.copy()
        clv_results['predicted_purchases'] = predicted_purchases
        clv_results['predicted_avg_value'] = predicted_avg_value
        clv_results['predicted_clv'] = predicted_clv
        clv_results['prob_alive'] = prob_alive
        clv_results['prediction_period_days'] = prediction_period_days
        
        # Add customer segments based on CLV
        clv_results['clv_segment'] = pd.qcut(
            clv_results['predicted_clv'], 
            q=5, 
            labels=['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value']
        )
        
        self.clv_predictions = clv_results
        
        # Save predictions
        clv_results.to_csv('data/processed/clv_predictions.csv', index=False)
        
        print(f"CLV predictions completed for {len(clv_results)} customers")
        print(f"Average predicted CLV: ${clv_results['predicted_clv'].mean():.2f}")
        print(f"Median predicted CLV: ${clv_results['predicted_clv'].median():.2f}")
        
        return clv_results
    
    def visualize_clv_analysis(self):
        """Create comprehensive CLV visualizations"""
        if self.clv_predictions is None:
            self.predict_clv()
        
        # Set up plotting
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Customer Lifetime Value Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. CLV Distribution
        axes[0, 0].hist(self.clv_predictions['predicted_clv'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('CLV Distribution')
        axes[0, 0].set_xlabel('Predicted CLV ($)')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].axvline(self.clv_predictions['predicted_clv'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(self.clv_predictions['predicted_clv'].median(), color='orange', linestyle='--', label='Median')
        axes[0, 0].legend()
        
        # 2. CLV vs Frequency
        axes[0, 1].scatter(self.clv_predictions['frequency'], self.clv_predictions['predicted_clv'], alpha=0.6)
        axes[0, 1].set_title('CLV vs Purchase Frequency')
        axes[0, 1].set_xlabel('Historical Frequency')
        axes[0, 1].set_ylabel('Predicted CLV ($)')
        
        # 3. CLV vs Monetary Value
        axes[0, 2].scatter(self.clv_predictions['monetary_value'], self.clv_predictions['predicted_clv'], alpha=0.6)
        axes[0, 2].set_title('CLV vs Historical Monetary Value')
        axes[0, 2].set_xlabel('Historical Avg Transaction ($)')
        axes[0, 2].set_ylabel('Predicted CLV ($)')
        
        # 4. CLV Segments
        segment_counts = self.clv_predictions['clv_segment'].value_counts()
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('CLV Segments Distribution')
        
        # 5. Probability Alive Distribution
        axes[1, 1].hist(self.clv_predictions['prob_alive'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Probability Alive Distribution')
        axes[1, 1].set_xlabel('Probability of Being Alive')
        axes[1, 1].set_ylabel('Number of Customers')
        
        # 6. Predicted Purchases vs CLV
        axes[1, 2].scatter(self.clv_predictions['predicted_purchases'], self.clv_predictions['predicted_clv'], alpha=0.6)
        axes[1, 2].set_title('Predicted Purchases vs CLV')
        axes[1, 2].set_xlabel('Predicted Purchases (Next Year)')
        axes[1, 2].set_ylabel('Predicted CLV ($)')
        
        # 7. CLV by Segment (Box plot)
        segment_data = [self.clv_predictions[self.clv_predictions['clv_segment'] == segment]['predicted_clv'].values 
                       for segment in segment_counts.index]
        axes[2, 0].boxplot(segment_data, labels=segment_counts.index)
        axes[2, 0].set_title('CLV Distribution by Segment')
        axes[2, 0].set_ylabel('Predicted CLV ($)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Recency vs CLV
        axes[2, 1].scatter(self.clv_predictions['recency'], self.clv_predictions['predicted_clv'], alpha=0.6)
        axes[2, 1].set_title('Recency vs CLV')
        axes[2, 1].set_xlabel('Recency (Days)')
        axes[2, 1].set_ylabel('Predicted CLV ($)')
        
        # 9. Top 20 Customers by CLV
        top_customers = self.clv_predictions.nlargest(20, 'predicted_clv')
        axes[2, 2].barh(range(len(top_customers)), top_customers['predicted_clv'], color='gold')
        axes[2, 2].set_title('Top 20 Customers by CLV')
        axes[2, 2].set_xlabel('Predicted CLV ($)')
        axes[2, 2].set_ylabel('Customer Rank')
        axes[2, 2].set_yticks(range(len(top_customers)))
        axes[2, 2].set_yticklabels([f'Customer {int(cid)}' for cid in top_customers['CustomerID']])
        
        plt.tight_layout()
        plt.savefig('visualizations/clv_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create model diagnostic plots
        self.create_model_diagnostics()
    
    def create_model_diagnostics(self):
        """Create model diagnostic visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CLV Model Diagnostics', fontsize=16, fontweight='bold')
        
        # 1. Frequency/Recency Matrix
        plot_frequency_recency_matrix(self.bgf, ax=axes[0, 0])
        axes[0, 0].set_title('Expected Purchases (BG/NBD Model)')
        
        # 2. Probability Alive Matrix
        plot_probability_alive_matrix(self.bgf, ax=axes[0, 1])
        axes[0, 1].set_title('Probability Customer is Alive')
        
        # 3. Model fit comparison (actual vs predicted frequency)
        from lifetimes.plotting import plot_period_transactions
        plot_period_transactions(self.bgf, ax=axes[1, 0])
        axes[1, 0].set_title('Model Fit: Actual vs Predicted Transactions')
        
        # 4. Calibration plot
        from lifetimes.plotting import plot_calibration_curve
        plot_calibration_curve(self.bgf, self.summary_df, kind='frequency', ax=axes[1, 1])
        axes[1, 1].set_title('Calibration: Frequency')
        
        plt.tight_layout()
        plt.savefig('visualizations/clv_model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_business_insights(self):
        """Generate actionable business insights from CLV analysis"""
        if self.clv_predictions is None:
            self.predict_clv()
        
        insights = {}
        
        # Overall CLV insights
        total_predicted_clv = self.clv_predictions['predicted_clv'].sum()
        avg_clv = self.clv_predictions['predicted_clv'].mean()
        median_clv = self.clv_predictions['predicted_clv'].median()
        
        insights['overall'] = {
            'total_predicted_clv': total_predicted_clv,
            'average_clv': avg_clv,
            'median_clv': median_clv,
            'total_customers': len(self.clv_predictions)
        }
        
        # Segment insights
        segment_analysis = self.clv_predictions.groupby('clv_segment').agg({
            'CustomerID': 'count',
            'predicted_clv': ['sum', 'mean'],
            'prob_alive': 'mean',
            'frequency': 'mean',
            'monetary_value': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['customer_count', 'total_clv', 'avg_clv', 'avg_prob_alive', 'avg_frequency', 'avg_monetary']
        segment_analysis['clv_percentage'] = (segment_analysis['total_clv'] / total_predicted_clv * 100).round(2)
        
        insights['segments'] = segment_analysis.to_dict('index')
        
        # Top customers
        top_10_customers = self.clv_predictions.nlargest(10, 'predicted_clv')[['CustomerID', 'predicted_clv', 'prob_alive']]
        insights['top_customers'] = top_10_customers.to_dict('records')
        
        # At-risk high-value customers
        at_risk_threshold = 0.3  # Probability alive < 30%
        high_value_threshold = self.clv_predictions['predicted_clv'].quantile(0.8)
        
        at_risk_high_value = self.clv_predictions[
            (self.clv_predictions['prob_alive'] < at_risk_threshold) &
            (self.clv_predictions['predicted_clv'] > high_value_threshold)
        ][['CustomerID', 'predicted_clv', 'prob_alive']]
        
        insights['at_risk_high_value'] = at_risk_high_value.to_dict('records')
        
        # Business recommendations
        insights['recommendations'] = {
            'high_value_customers': f"Focus retention efforts on {len(top_10_customers)} top customers representing ${top_10_customers['predicted_clv'].sum():.0f} in predicted value",
            'at_risk_customers': f"Immediate intervention needed for {len(at_risk_high_value)} high-value at-risk customers",
            'acquisition_benchmark': f"Target new customer acquisition with CLV > ${avg_clv:.0f} (current average)",
            'retention_priority': "Prioritize customers in 'Above Average' and 'High Value' segments for retention campaigns"
        }
        
        return insights

if __name__ == "__main__":
    # Initialize CLV predictor
    clv_predictor = CLVPredictor()
    
    # Prepare data and fit models
    summary_data = clv_predictor.prepare_clv_data()
    bgf_model = clv_predictor.fit_bgf_model()
    ggf_model = clv_predictor.fit_gamma_gamma_model()
    
    # Predict CLV
    clv_results = clv_predictor.predict_clv(prediction_period_days=365)
    
    # Create visualizations
    clv_predictor.visualize_clv_analysis()
    
    # Generate business insights
    business_insights = clv_predictor.generate_business_insights()
    
    print("\nCLV Analysis completed successfully!")
    print("Files generated:")
    print("- data/processed/clv_summary_data.csv")
    print("- data/processed/clv_predictions.csv")
    print("- visualizations/clv_analysis_dashboard.png")
    print("- visualizations/clv_model_diagnostics.png")