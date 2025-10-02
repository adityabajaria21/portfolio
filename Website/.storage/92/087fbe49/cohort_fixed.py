"""
Customer Retention & Cohort Analysis System (Fixed)
Advanced cohort analysis for customer retention insights
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CohortAnalysisSystem:
    def __init__(self):
        self.cohort_data = None
        self.retention_table = None
        
    def generate_customer_data(self, n_customers=5000):
        """Generate realistic customer transaction dataset"""
        np.random.seed(42)
        
        # Generate customers with acquisition dates
        start_date = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        customers = []
        for i in range(n_customers):
            acquisition_date = np.random.choice(start_date)
            customer_id = f'C{i+1:05d}'
            
            # Generate transactions for this customer
            transactions = []
            current_date = acquisition_date
            
            # Customer lifetime (some churn, some stay active)
            lifetime_days = np.random.exponential(180)  # Average 6 months
            end_date = pd.Timestamp('2024-01-01')
            max_lifetime = (end_date - acquisition_date).days
            lifetime_days = min(lifetime_days, max_lifetime)
            
            # Generate transactions within lifetime
            transaction_count = 0
            days_elapsed = 0
            
            while days_elapsed < lifetime_days:
                # Probability of transaction decreases over time (churn)
                transaction_prob = 0.8 * np.exp(-days_elapsed / 200)  # Exponential decay
                
                if np.random.random() < transaction_prob:
                    transaction_count += 1
                    transactions.append({
                        'customer_id': customer_id,
                        'transaction_date': current_date,
                        'transaction_id': f'T{customer_id}_{transaction_count:03d}',
                        'amount': np.random.lognormal(4, 0.8),  # Transaction amount
                        'acquisition_date': acquisition_date
                    })
                
                # Next potential transaction (weekly pattern)
                days_to_add = np.random.poisson(7)
                current_date += pd.Timedelta(days=days_to_add)
                days_elapsed += days_to_add
            
            customers.extend(transactions)
        
        self.data = pd.DataFrame(customers)
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        self.data['acquisition_date'] = pd.to_datetime(self.data['acquisition_date'])
        
        print(f"✅ Generated {len(self.data):,} transactions for {n_customers:,} customers")
        print(f"   - Date range: {self.data['transaction_date'].min()} to {self.data['transaction_date'].max()}")
        print(f"   - Average transactions per customer: {len(self.data) / n_customers:.1f}")
        print(f"   - Total revenue: ${self.data['amount'].sum():,.2f}")
        
        return self.data
    
    def create_cohort_analysis(self):
        """Create cohort analysis with retention rates"""
        print("🔄 Creating cohort analysis...")
        
        # Create period columns
        self.data['acquisition_period'] = self.data['acquisition_date'].dt.to_period('M')
        self.data['transaction_period'] = self.data['transaction_date'].dt.to_period('M')
        
        # Calculate period number (months since acquisition)
        period_diffs = []
        for idx, row in self.data.iterrows():
            diff = (row['transaction_period'].ordinal - row['acquisition_period'].ordinal)
            period_diffs.append(diff)
        
        self.data['period_number'] = period_diffs
        
        # Create cohort table
        cohort_data = self.data.groupby(['acquisition_period', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_data.rename(columns={'customer_id': 'customers'}, inplace=True)
        
        # Pivot to create cohort table
        cohort_table = cohort_data.pivot(index='acquisition_period', 
                                        columns='period_number', 
                                        values='customers')
        
        # Get cohort sizes (customers acquired in each month)
        cohort_sizes = self.data.groupby('acquisition_period')['customer_id'].nunique()
        
        # Calculate retention rates
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        self.cohort_table = cohort_table
        self.retention_table = retention_table
        self.cohort_sizes = cohort_sizes
        
        print(f"✅ Cohort analysis complete!")
        print(f"   - Cohorts analyzed: {len(cohort_sizes)}")
        print(f"   - Max retention period: {cohort_table.columns.max()} months")
        if 1 in retention_table.columns:
            print(f"   - Average month 1 retention: {retention_table[1].mean():.2%}")
    
    def run_complete_analysis(self):
        """Run complete cohort analysis"""
        print("🚀 Starting Customer Retention & Cohort Analysis...")
        
        # Generate and analyze data
        self.generate_customer_data()
        self.create_cohort_analysis()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/customer_transaction_data.csv', index=False)
        self.retention_table.to_csv('results/cohort_retention_table.csv')
        
        print("\n✅ Cohort Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/customer_transaction_data.csv")
        print("   - results/cohort_retention_table.csv")
        
        month_1_retention = self.retention_table[1].mean() if 1 in self.retention_table.columns else 0
        
        return {
            'status': 'complete',
            'customers': self.data['customer_id'].nunique(),
            'retention_month_1': f"{month_1_retention:.1%}"
        }

if __name__ == "__main__":
    cohort_system = CohortAnalysisSystem()
    results = cohort_system.run_complete_analysis()