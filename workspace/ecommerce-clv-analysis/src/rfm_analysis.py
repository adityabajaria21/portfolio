"""
RFM Analysis Module
Implements Recency, Frequency, and Monetary analysis for customer segmentation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class RFMAnalyzer:
    def __init__(self, data_path='data/processed/cleaned_data.csv'):
        self.data_path = data_path
        self.df = None
        self.rfm_df = None
        self.reference_date = None
        
    def load_data(self):
        """Load cleaned data"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            print(f"Data loaded: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("Cleaned data not found. Please run data preprocessing first.")
            return None
    
    def calculate_rfm(self):
        """Calculate RFM metrics for each customer"""
        if self.df is None:
            self.load_data()
        
        # Set reference date (day after the last transaction)
        self.reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        print(f"Reference date for recency calculation: {self.reference_date}")
        
        # Calculate RFM metrics
        rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Remove customers with negative monetary values
        rfm = rfm[rfm['Monetary'] > 0]
        
        print(f"RFM calculated for {len(rfm)} customers")
        print(f"RFM Summary:")
        print(rfm.describe())
        
        self.rfm_df = rfm
        return rfm
    
    def calculate_rfm_scores(self, recency_bins=5, frequency_bins=5, monetary_bins=5):
        """Calculate RFM scores using quintile-based scoring"""
        if self.rfm_df is None:
            self.calculate_rfm()
        
        rfm_scores = self.rfm_df.copy()
        
        # Calculate quintiles for each metric
        # Note: For recency, lower values are better (more recent), so we reverse the scoring
        rfm_scores['R_Score'] = pd.qcut(rfm_scores['Recency'], recency_bins, labels=range(recency_bins, 0, -1))
        rfm_scores['F_Score'] = pd.qcut(rfm_scores['Frequency'].rank(method='first'), frequency_bins, labels=range(1, frequency_bins + 1))
        rfm_scores['M_Score'] = pd.qcut(rfm_scores['Monetary'], monetary_bins, labels=range(1, monetary_bins + 1))
        
        # Convert to numeric
        rfm_scores['R_Score'] = rfm_scores['R_Score'].astype(int)
        rfm_scores['F_Score'] = rfm_scores['F_Score'].astype(int)
        rfm_scores['M_Score'] = rfm_scores['M_Score'].astype(int)
        
        # Calculate overall RFM score
        rfm_scores['RFM_Score'] = rfm_scores['R_Score'].astype(str) + rfm_scores['F_Score'].astype(str) + rfm_scores['M_Score'].astype(str)
        
        # Calculate weighted RFM score
        rfm_scores['RFM_Score_Weighted'] = (rfm_scores['R_Score'] * 0.3 + 
                                          rfm_scores['F_Score'] * 0.4 + 
                                          rfm_scores['M_Score'] * 0.3)
        
        self.rfm_df = rfm_scores
        
        # Save RFM results
        rfm_scores.to_csv('data/processed/rfm_analysis.csv', index=False)
        print("RFM scores calculated and saved to data/processed/rfm_analysis.csv")
        
        return rfm_scores
    
    def segment_customers(self):
        """Segment customers based on RFM scores"""
        if 'RFM_Score_Weighted' not in self.rfm_df.columns:
            self.calculate_rfm_scores()
        
        def rfm_segment(row):
            """Define customer segments based on RFM scores"""
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Potential Loyalists'
            elif r >= 3 and f <= 2 and m <= 2:
                return 'Promising'
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cannot Lose Them'
            elif r <= 2 and f >= 2 and m >= 2:
                return 'At Risk'
            elif r <= 2 and f >= 4 and m <= 2:
                return 'Hibernating'
            else:
                return 'Lost'
        
        self.rfm_df['Segment'] = self.rfm_df.apply(rfm_segment, axis=1)
        
        # Calculate segment statistics
        segment_stats = self.rfm_df.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'RFM_Score_Weighted': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_RFM_Score']
        segment_stats['Percentage'] = (segment_stats['Count'] / len(self.rfm_df) * 100).round(2)
        
        print("\nCustomer Segments:")
        print(segment_stats)
        
        # Save segmented data
        self.rfm_df.to_csv('data/processed/customer_segments.csv', index=False)
        segment_stats.to_csv('data/processed/segment_statistics.csv')
        
        return self.rfm_df, segment_stats
    
    def visualize_rfm(self):
        """Create visualizations for RFM analysis"""
        if self.rfm_df is None or 'Segment' not in self.rfm_df.columns:
            self.segment_customers()
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. RFM Distribution
        axes[0, 0].hist(self.rfm_df['Recency'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        
        axes[0, 1].hist(self.rfm_df['Frequency'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        axes[0, 2].hist(self.rfm_df['Monetary'], bins=50, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spent ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        # 2. Customer Segments
        segment_counts = self.rfm_df['Segment'].value_counts()
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Customer Segments Distribution')
        
        # 3. RFM Score Distribution
        axes[1, 1].hist(self.rfm_df['RFM_Score_Weighted'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('RFM Score Distribution')
        axes[1, 1].set_xlabel('Weighted RFM Score')
        axes[1, 1].set_ylabel('Number of Customers')
        
        # 4. Segment vs Monetary Value
        segment_monetary = self.rfm_df.groupby('Segment')['Monetary'].mean().sort_values(ascending=True)
        axes[1, 2].barh(range(len(segment_monetary)), segment_monetary.values, color='lightcoral')
        axes[1, 2].set_yticks(range(len(segment_monetary)))
        axes[1, 2].set_yticklabels(segment_monetary.index)
        axes[1, 2].set_title('Average Monetary Value by Segment')
        axes[1, 2].set_xlabel('Average Monetary Value ($)')
        
        plt.tight_layout()
        plt.savefig('visualizations/rfm_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.rfm_df[['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('RFM Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/rfm_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_segment_insights(self):
        """Generate insights and recommendations for each segment"""
        if 'Segment' not in self.rfm_df.columns:
            self.segment_customers()
        
        insights = {
            'Champions': {
                'description': 'Best customers who bought recently, buy often and spend the most',
                'marketing_strategy': 'Reward them, ask for reviews, upsell premium products',
                'characteristics': 'High value, high engagement'
            },
            'Loyal Customers': {
                'description': 'Customers who buy regularly and have good monetary value',
                'marketing_strategy': 'Keep them engaged with loyalty programs and exclusive offers',
                'characteristics': 'Consistent purchasers'
            },
            'New Customers': {
                'description': 'Recent customers with potential for growth',
                'marketing_strategy': 'Provide onboarding support and encourage repeat purchases',
                'characteristics': 'Recent but low frequency'
            },
            'Potential Loyalists': {
                'description': 'Recent customers with good purchase frequency but lower monetary value',
                'marketing_strategy': 'Offer membership programs and personalized recommendations',
                'characteristics': 'Good engagement, room for monetary growth'
            },
            'At Risk': {
                'description': 'Customers who haven\'t purchased recently but were valuable',
                'marketing_strategy': 'Send personalized reactivation campaigns and special offers',
                'characteristics': 'Declining engagement'
            },
            'Cannot Lose Them': {
                'description': 'High-value customers who haven\'t purchased recently',
                'marketing_strategy': 'Urgent intervention with personalized outreach and exclusive deals',
                'characteristics': 'High value but at risk of churning'
            },
            'Hibernating': {
                'description': 'Customers who were frequent buyers but haven\'t purchased recently',
                'marketing_strategy': 'Re-engagement campaigns with product recommendations',
                'characteristics': 'Previously engaged but now dormant'
            },
            'Lost': {
                'description': 'Customers with low scores across all RFM metrics',
                'marketing_strategy': 'Minimal investment, focus on win-back campaigns if cost-effective',
                'characteristics': 'Low value and engagement'
            }
        }
        
        return insights

if __name__ == "__main__":
    # Initialize RFM analyzer
    rfm_analyzer = RFMAnalyzer()
    
    # Perform RFM analysis
    rfm_data = rfm_analyzer.calculate_rfm_scores()
    segmented_data, segment_stats = rfm_analyzer.segment_customers()
    
    # Create visualizations
    rfm_analyzer.visualize_rfm()
    
    # Get insights
    insights = rfm_analyzer.get_segment_insights()
    
    print("\nRFM Analysis completed successfully!")
    print("Files generated:")
    print("- data/processed/rfm_analysis.csv")
    print("- data/processed/customer_segments.csv")
    print("- data/processed/segment_statistics.csv")
    print("- visualizations/rfm_analysis_dashboard.png")
    print("- visualizations/rfm_correlation_matrix.png")