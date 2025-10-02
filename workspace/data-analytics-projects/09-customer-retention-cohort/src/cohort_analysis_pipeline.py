"""
Customer Retention & Cohort Analysis System
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
            lifetime_days = min(lifetime_days, (pd.Timestamp('2024-01-01') - acquisition_date).days)
            
            # Generate transactions within lifetime
            transaction_count = 0
            while (current_date - acquisition_date).days < lifetime_days:
                # Probability of transaction decreases over time (churn)
                days_since_acquisition = (current_date - acquisition_date).days
                transaction_prob = 0.8 * np.exp(-days_since_acquisition / 200)  # Exponential decay
                
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
                current_date += pd.Timedelta(days=np.random.poisson(7))
            
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
        self.data['period_number'] = (
            self.data['transaction_period'] - self.data['acquisition_period']
        ).apply(attrgetter('n'))
        
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
        print(f"   - Average month 1 retention: {retention_table[1].mean():.2%}")
    
    def calculate_ltv_by_cohort(self):
        """Calculate customer lifetime value by cohort"""
        print("🔄 Calculating LTV by cohort...")
        
        # Calculate revenue by cohort and period
        revenue_data = self.data.groupby(['acquisition_period', 'period_number'])['amount'].sum().reset_index()
        revenue_table = revenue_data.pivot(index='acquisition_period', 
                                         columns='period_number', 
                                         values='amount')
        
        # Calculate cumulative LTV
        cumulative_ltv = revenue_table.cumsum(axis=1).divide(self.cohort_sizes, axis=0)
        
        # Calculate average LTV by period
        avg_ltv_by_period = cumulative_ltv.mean()
        
        self.revenue_table = revenue_table
        self.cumulative_ltv = cumulative_ltv
        self.avg_ltv_by_period = avg_ltv_by_period
        
        print(f"✅ LTV analysis complete!")
        print(f"   - Average LTV (Month 6): ${avg_ltv_by_period.get(6, 0):.2f}")
        print(f"   - Average LTV (Month 12): ${avg_ltv_by_period.get(12, 0):.2f}")
    
    def create_cohort_dashboard(self):
        """Create comprehensive cohort analysis dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Cohort Retention Heatmap', 'Retention Curves', 'Cohort Sizes',
                          'LTV by Cohort', 'Average LTV Growth', 'Churn Analysis',
                          'Monthly Revenue Trends', 'Customer Acquisition', 'Retention Benchmarks'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Cohort Retention Heatmap
        fig.add_trace(
            go.Heatmap(
                z=self.retention_table.values,
                x=self.retention_table.columns,
                y=[str(idx) for idx in self.retention_table.index],
                colorscale='RdYlBu_r',
                name='Retention Rate'
            ),
            row=1, col=1
        )
        
        # 2. Retention Curves
        for i, cohort in enumerate(self.retention_table.index[-6:]):  # Last 6 cohorts
            retention_curve = self.retention_table.loc[cohort].dropna()
            fig.add_trace(
                go.Scatter(x=retention_curve.index, y=retention_curve.values,
                          mode='lines+markers', name=f'Cohort {cohort}'),
                row=1, col=2
            )
        
        # 3. Cohort Sizes
        fig.add_trace(
            go.Bar(x=[str(idx) for idx in self.cohort_sizes.index], 
                   y=self.cohort_sizes.values, name='Cohort Size'),
            row=1, col=3
        )
        
        # 4. LTV by Cohort Heatmap
        fig.add_trace(
            go.Heatmap(
                z=self.cumulative_ltv.values,
                x=self.cumulative_ltv.columns,
                y=[str(idx) for idx in self.cumulative_ltv.index],
                colorscale='Viridis',
                name='Cumulative LTV'
            ),
            row=2, col=1
        )
        
        # 5. Average LTV Growth
        fig.add_trace(
            go.Scatter(x=self.avg_ltv_by_period.index, y=self.avg_ltv_by_period.values,
                      mode='lines+markers', name='Average LTV'),
            row=2, col=2
        )
        
        # 6. Churn Analysis (1 - retention rate)
        churn_rates = 1 - self.retention_table.mean()
        fig.add_trace(
            go.Bar(x=churn_rates.index, y=churn_rates.values, name='Churn Rate'),
            row=2, col=3
        )
        
        # 7. Monthly Revenue Trends
        monthly_revenue = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['amount'].sum()
        fig.add_trace(
            go.Scatter(x=[str(idx) for idx in monthly_revenue.index], y=monthly_revenue.values,
                      mode='lines+markers', name='Monthly Revenue'),
            row=3, col=1
        )
        
        # 8. Customer Acquisition by Month
        monthly_acquisitions = self.data.groupby('acquisition_period')['customer_id'].nunique()
        fig.add_trace(
            go.Bar(x=[str(idx) for idx in monthly_acquisitions.index], 
                   y=monthly_acquisitions.values, name='New Customers'),
            row=3, col=2
        )
        
        # 9. Retention Benchmarks Table
        benchmarks = pd.DataFrame({
            'Period': ['Month 1', 'Month 3', 'Month 6', 'Month 12'],
            'Retention Rate': [
                f"{self.retention_table[1].mean():.1%}",
                f"{self.retention_table[3].mean():.1%}" if 3 in self.retention_table.columns else "N/A",
                f"{self.retention_table[6].mean():.1%}" if 6 in self.retention_table.columns else "N/A",
                f"{self.retention_table[12].mean():.1%}" if 12 in self.retention_table.columns else "N/A"
            ],
            'Benchmark': ['70-80%', '40-50%', '25-35%', '15-25%']
        })
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(benchmarks.columns)),
                cells=dict(values=[benchmarks[col] for col in benchmarks.columns])
            ),
            row=3, col=3
        )
        
        fig.update_layout(height=1200, title_text="Customer Cohort Analysis Dashboard")
        return fig
    
    def generate_retention_insights(self):
        """Generate actionable retention insights and recommendations"""
        # Calculate key metrics
        month_1_retention = self.retention_table[1].mean()
        month_3_retention = self.retention_table[3].mean() if 3 in self.retention_table.columns else 0
        month_6_retention = self.retention_table[6].mean() if 6 in self.retention_table.columns else 0
        
        # Best and worst performing cohorts
        best_cohort = self.retention_table[1].idxmax()
        worst_cohort = self.retention_table[1].idxmin()
        
        # LTV insights
        avg_ltv_6m = self.avg_ltv_by_period.get(6, 0)
        avg_ltv_12m = self.avg_ltv_by_period.get(12, 0)
        
        insights = {
            'retention_summary': {
                'month_1_retention': f"{month_1_retention:.2%}",
                'month_3_retention': f"{month_3_retention:.2%}",
                'month_6_retention': f"{month_6_retention:.2%}",
                'best_performing_cohort': str(best_cohort),
                'worst_performing_cohort': str(worst_cohort)
            },
            'ltv_insights': {
                'average_ltv_6_months': f"${avg_ltv_6m:.2f}",
                'average_ltv_12_months': f"${avg_ltv_12m:.2f}",
                'ltv_growth_rate': f"{((avg_ltv_12m / avg_ltv_6m - 1) * 100):.1f}%" if avg_ltv_6m > 0 else "N/A"
            },
            'immediate_actions': [
                f"Focus on improving Month 1 retention (currently {month_1_retention:.1%})",
                f"Investigate why {best_cohort} cohort performed best",
                f"Address issues affecting {worst_cohort} cohort",
                "Implement onboarding improvements for new customers"
            ],
            'retention_strategies': [
                "Develop targeted re-engagement campaigns for Month 2-3 customers",
                "Create loyalty programs for customers past Month 6",
                "Implement personalized product recommendations",
                "Set up automated email sequences for different cohort stages"
            ],
            'ltv_optimization': [
                f"Focus on increasing LTV beyond ${avg_ltv_6m:.0f} in first 6 months",
                "Develop upselling strategies for retained customers",
                "Create subscription or membership programs",
                "Implement referral programs to leverage satisfied customers"
            ],
            'monitoring_metrics': [
                "Weekly cohort retention rates",
                "Monthly LTV progression",
                "Customer acquisition cost vs LTV ratio",
                "Churn prediction model accuracy"
            ]
        }
        
        return insights
    
    def run_complete_analysis(self):
        """Run complete cohort analysis"""
        print("🚀 Starting Customer Retention & Cohort Analysis...")
        
        # Generate and analyze data
        self.generate_customer_data()
        self.create_cohort_analysis()
        self.calculate_ltv_by_cohort()
        
        # Create visualizations
        dashboard = self.create_cohort_dashboard()
        
        # Generate insights
        insights = self.generate_retention_insights()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        dashboard.write_html('results/cohort_analysis_dashboard.html')
        self.data.to_csv('results/customer_transaction_data.csv', index=False)
        self.retention_table.to_csv('results/cohort_retention_table.csv')
        self.cumulative_ltv.to_csv('results/cohort_ltv_table.csv')
        
        import json
        with open('results/retention_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'dataset_summary': {
                'total_customers': self.data['customer_id'].nunique(),
                'total_transactions': len(self.data),
                'total_revenue': f"${self.data['amount'].sum():,.2f}",
                'analysis_period': f"{self.data['acquisition_date'].min()} to {self.data['transaction_date'].max()}"
            },
            'cohort_analysis': {
                'cohorts_analyzed': len(self.cohort_sizes),
                'month_1_retention': f"{self.retention_table[1].mean():.2%}",
                'month_6_retention': f"{self.retention_table[6].mean():.2%}" if 6 in self.retention_table.columns else "N/A",
                'average_ltv_6m': f"${self.avg_ltv_by_period.get(6, 0):.2f}"
            },
            'key_insights': [
                f"Month 1 retention rate: {self.retention_table[1].mean():.1%}",
                f"Best performing cohort: {self.retention_table[1].idxmax()}",
                f"Average customer LTV: ${self.avg_ltv_by_period.get(6, 0):.2f}",
                f"Total cohorts tracked: {len(self.cohort_sizes)}"
            ]
        }
        
        with open('results/cohort_analysis_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print("\n✅ Cohort Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/cohort_analysis_dashboard.html")
        print("   - results/customer_transaction_data.csv")
        print("   - results/cohort_retention_table.csv")
        print("   - results/cohort_ltv_table.csv")
        print("   - results/retention_insights.json")
        print("   - results/cohort_analysis_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Customers Analyzed: {self.data['customer_id'].nunique():,}")
        print(f"   - Cohorts Tracked: {len(self.cohort_sizes)}")
        print(f"   - Month 1 Retention: {self.retention_table[1].mean():.1%}")
        print(f"   - Average LTV: ${self.avg_ltv_by_period.get(6, 0):.2f}")
        
        return summary_report

# Fix import issue
from operator import attrgetter

if __name__ == "__main__":
    cohort_system = CohortAnalysisSystem()
    results = cohort_system.run_complete_analysis()