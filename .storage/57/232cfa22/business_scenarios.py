"""
Additional Business Scenario Visualizations
Creates specific visualizations for different business use cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class BusinessScenarioVisualizer:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load processed data files"""
        try:
            self.clv_data = pd.read_csv('data/processed/clv_predictions.csv')
            self.rfm_data = pd.read_csv('data/processed/customer_segments.csv')
            print("✅ Data loaded successfully")
        except FileNotFoundError:
            print("⚠️ Data files not found, generating sample data...")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data if files don't exist"""
        np.random.seed(42)
        n_customers = 2000
        
        # CLV Data
        self.clv_data = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'predicted_clv': np.random.lognormal(6, 1, n_customers),
            'predicted_purchases': np.random.poisson(3, n_customers),
            'prob_alive': np.random.beta(2, 1, n_customers),
            'clv_segment': np.random.choice(['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value'], n_customers)
        })
        
        # RFM Data
        self.rfm_data = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'Recency': np.random.exponential(50, n_customers),
            'Frequency': np.random.poisson(5, n_customers) + 1,
            'Monetary': np.random.lognormal(5, 1, n_customers),
            'Segment': np.random.choice(['Champions', 'Loyal Customers', 'At Risk', 'New Customers', 'Others'], n_customers)
        })
        
        # Save sample data
        os.makedirs('data/processed', exist_ok=True)
        self.clv_data.to_csv('data/processed/clv_predictions.csv', index=False)
        self.rfm_data.to_csv('data/processed/customer_segments.csv', index=False)
    
    def create_retention_campaign_analysis(self):
        """Create visualizations for retention campaign targeting"""
        print("📊 Creating Retention Campaign Analysis...")
        
        # Identify at-risk high-value customers
        high_value_threshold = self.clv_data['predicted_clv'].quantile(0.8)
        at_risk_threshold = 0.3
        
        self.clv_data['campaign_priority'] = 'Low Priority'
        self.clv_data.loc[
            (self.clv_data['predicted_clv'] > high_value_threshold) & 
            (self.clv_data['prob_alive'] < at_risk_threshold), 
            'campaign_priority'
        ] = 'Urgent - High Value At Risk'
        
        self.clv_data.loc[
            (self.clv_data['predicted_clv'] > high_value_threshold) & 
            (self.clv_data['prob_alive'] >= at_risk_threshold), 
            'campaign_priority'
        ] = 'High Value - Monitor'
        
        self.clv_data.loc[
            (self.clv_data['predicted_clv'] <= high_value_threshold) & 
            (self.clv_data['prob_alive'] < at_risk_threshold), 
            'campaign_priority'
        ] = 'Medium Priority'
        
        # Create retention campaign dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Campaign Priority Distribution', 'CLV vs Churn Risk', 
                          'High-Value At-Risk Customers', 'Campaign Budget Allocation'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Priority distribution
        priority_counts = self.clv_data['campaign_priority'].value_counts()
        fig.add_trace(
            go.Pie(labels=priority_counts.index, values=priority_counts.values, name="Priority"),
            row=1, col=1
        )
        
        # CLV vs Churn Risk scatter
        fig.add_trace(
            go.Scatter(
                x=self.clv_data['prob_alive'],
                y=self.clv_data['predicted_clv'],
                mode='markers',
                marker=dict(
                    color=self.clv_data['predicted_clv'],
                    colorscale='Viridis',
                    size=8,
                    opacity=0.6
                ),
                name="Customers"
            ),
            row=1, col=2
        )
        
        # High-value at-risk customers
        urgent_customers = self.clv_data[self.clv_data['campaign_priority'] == 'Urgent - High Value At Risk']
        top_urgent = urgent_customers.nlargest(10, 'predicted_clv')
        
        fig.add_trace(
            go.Bar(
                x=top_urgent['CustomerID'].astype(str),
                y=top_urgent['predicted_clv'],
                name="At-Risk CLV",
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # Campaign budget allocation
        budget_allocation = {
            'Urgent - High Value At Risk': 50,
            'High Value - Monitor': 30,
            'Medium Priority': 15,
            'Low Priority': 5
        }
        
        fig.add_trace(
            go.Bar(
                x=list(budget_allocation.keys()),
                y=list(budget_allocation.values()),
                name="Budget %",
                marker_color=['red', 'orange', 'yellow', 'green']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Customer Retention Campaign Analysis",
            height=800,
            showlegend=False
        )
        
        fig.write_html('visualizations/retention_campaign_analysis.html')
        print("✅ Retention campaign analysis saved to visualizations/retention_campaign_analysis.html")
        
        return fig
    
    def create_revenue_optimization_dashboard(self):
        """Create revenue optimization scenarios"""
        print("📊 Creating Revenue Optimization Dashboard...")
        
        # Calculate revenue scenarios
        current_revenue = self.clv_data['predicted_clv'].sum()
        
        # Scenario 1: 10% improvement in high-value retention
        high_value_mask = self.clv_data['clv_segment'] == 'High Value'
        scenario1_revenue = current_revenue + (self.clv_data[high_value_mask]['predicted_clv'].sum() * 0.1)
        
        # Scenario 2: Convert 20% of 'Above Average' to 'High Value'
        above_avg_mask = self.clv_data['clv_segment'] == 'Above Average'
        conversion_boost = self.clv_data[above_avg_mask]['predicted_clv'].mean() * 0.5  # 50% CLV boost
        scenario2_revenue = current_revenue + (len(self.clv_data[above_avg_mask]) * 0.2 * conversion_boost)
        
        # Scenario 3: Reduce churn by 25% for at-risk customers
        at_risk_customers = self.clv_data[self.clv_data['prob_alive'] < 0.5]
        scenario3_revenue = current_revenue + (at_risk_customers['predicted_clv'].sum() * 0.25)
        
        # Create revenue optimization dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Scenarios Comparison', 'CLV Distribution by Segment', 
                          'Monthly Revenue Projection', 'ROI Analysis'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Revenue scenarios
        scenarios = ['Current', 'High-Value Retention', 'Segment Conversion', 'Churn Reduction']
        revenues = [current_revenue, scenario1_revenue, scenario2_revenue, scenario3_revenue]
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=revenues,
                name="Revenue",
                marker_color=['blue', 'green', 'orange', 'purple']
            ),
            row=1, col=1
        )
        
        # CLV distribution by segment
        for segment in self.clv_data['clv_segment'].unique():
            segment_data = self.clv_data[self.clv_data['clv_segment'] == segment]
            fig.add_trace(
                go.Box(
                    y=segment_data['predicted_clv'],
                    name=segment,
                    boxmean=True
                ),
                row=1, col=2
            )
        
        # Monthly revenue projection (12 months)
        months = list(range(1, 13))
        monthly_revenue = [current_revenue / 12 * (1 + 0.02 * i) for i in months]  # 2% monthly growth
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=monthly_revenue,
                mode='lines+markers',
                name="Projected Revenue",
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
        
        # ROI Analysis
        investment_costs = [0, 50000, 75000, 40000]  # Investment for each scenario
        roi_percentages = [(rev - current_revenue - cost) / cost * 100 if cost > 0 else 0 
                          for rev, cost in zip(revenues, investment_costs)]
        
        fig.add_trace(
            go.Bar(
                x=scenarios[1:],  # Exclude current
                y=roi_percentages[1:],
                name="ROI %",
                marker_color=['green', 'orange', 'purple']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Revenue Optimization Scenarios",
            height=800,
            showlegend=True
        )
        
        fig.write_html('visualizations/revenue_optimization_dashboard.html')
        print("✅ Revenue optimization dashboard saved to visualizations/revenue_optimization_dashboard.html")
        
        return fig
    
    def create_marketing_budget_allocation(self):
        """Create marketing budget allocation analysis"""
        print("📊 Creating Marketing Budget Allocation Analysis...")
        
        # Calculate segment values and recommended budget allocation
        segment_stats = self.clv_data.groupby('clv_segment').agg({
            'predicted_clv': ['count', 'sum', 'mean'],
            'prob_alive': 'mean'
        }).round(2)
        
        segment_stats.columns = ['customer_count', 'total_clv', 'avg_clv', 'avg_prob_alive']
        segment_stats = segment_stats.reset_index()
        
        # Calculate recommended budget allocation based on CLV and risk
        total_clv = segment_stats['total_clv'].sum()
        segment_stats['clv_percentage'] = (segment_stats['total_clv'] / total_clv * 100).round(1)
        
        # Adjust budget allocation based on risk (lower prob_alive = higher budget need)
        risk_multiplier = 2 - segment_stats['avg_prob_alive']  # Higher risk = higher multiplier
        segment_stats['recommended_budget'] = (segment_stats['clv_percentage'] * risk_multiplier).round(1)
        
        # Normalize to 100%
        segment_stats['recommended_budget'] = (
            segment_stats['recommended_budget'] / segment_stats['recommended_budget'].sum() * 100
        ).round(1)
        
        # Create marketing budget dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Current CLV Distribution', 'Recommended Budget Allocation', 
                          'Budget vs CLV Efficiency', 'Segment Performance Matrix'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Current CLV distribution
        fig.add_trace(
            go.Pie(
                labels=segment_stats['clv_segment'],
                values=segment_stats['clv_percentage'],
                name="CLV Distribution"
            ),
            row=1, col=1
        )
        
        # Recommended budget allocation
        fig.add_trace(
            go.Pie(
                labels=segment_stats['clv_segment'],
                values=segment_stats['recommended_budget'],
                name="Budget Allocation"
            ),
            row=1, col=2
        )
        
        # Budget efficiency
        fig.add_trace(
            go.Bar(
                x=segment_stats['clv_segment'],
                y=segment_stats['recommended_budget'] / segment_stats['clv_percentage'],
                name="Budget Efficiency Ratio",
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Performance matrix
        fig.add_trace(
            go.Scatter(
                x=segment_stats['avg_clv'],
                y=segment_stats['avg_prob_alive'],
                mode='markers+text',
                text=segment_stats['clv_segment'],
                textposition="top center",
                marker=dict(
                    size=segment_stats['customer_count'] / 20,
                    color=segment_stats['recommended_budget'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name="Segments"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Marketing Budget Allocation Analysis",
            height=800,
            showlegend=False
        )
        
        fig.write_html('visualizations/marketing_budget_allocation.html')
        print("✅ Marketing budget allocation saved to visualizations/marketing_budget_allocation.html")
        
        # Save budget recommendations to CSV
        segment_stats.to_csv('data/processed/marketing_budget_recommendations.csv', index=False)
        
        return fig
    
    def create_customer_journey_analysis(self):
        """Create customer journey and lifecycle analysis"""
        print("📊 Creating Customer Journey Analysis...")
        
        # Simulate customer journey stages
        np.random.seed(42)
        n_customers = len(self.clv_data)
        
        journey_data = pd.DataFrame({
            'CustomerID': self.clv_data['CustomerID'],
            'acquisition_cost': np.random.uniform(20, 100, n_customers),
            'first_purchase_value': np.random.uniform(50, 500, n_customers),
            'time_to_second_purchase': np.random.exponential(30, n_customers),
            'total_purchases': np.random.poisson(5, n_customers) + 1,
            'customer_lifetime_days': np.random.uniform(30, 1000, n_customers)
        })
        
        # Merge with CLV data
        journey_analysis = pd.merge(journey_data, self.clv_data, on='CustomerID')
        
        # Calculate journey metrics
        journey_analysis['payback_period'] = (
            journey_analysis['acquisition_cost'] / 
            (journey_analysis['predicted_clv'] / 365)  # Daily CLV
        )
        
        journey_analysis['ltv_cac_ratio'] = (
            journey_analysis['predicted_clv'] / journey_analysis['acquisition_cost']
        )
        
        # Create customer journey dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('LTV/CAC Ratio Distribution', 'Payback Period Analysis', 
                          'Customer Lifecycle Stages', 'Journey Optimization Opportunities'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # LTV/CAC ratio distribution
        fig.add_trace(
            go.Histogram(
                x=journey_analysis['ltv_cac_ratio'],
                nbinsx=30,
                name="LTV/CAC Ratio"
            ),
            row=1, col=1
        )
        
        # Payback period by segment
        for segment in journey_analysis['clv_segment'].unique():
            segment_data = journey_analysis[journey_analysis['clv_segment'] == segment]
            fig.add_trace(
                go.Box(
                    y=segment_data['payback_period'],
                    name=segment,
                    boxmean=True
                ),
                row=1, col=2
            )
        
        # Lifecycle stages
        lifecycle_stages = ['New', 'Growing', 'Mature', 'At Risk', 'Lost']
        stage_counts = [400, 600, 500, 300, 200]  # Sample data
        
        fig.add_trace(
            go.Bar(
                x=lifecycle_stages,
                y=stage_counts,
                name="Customer Count",
                marker_color=['green', 'blue', 'orange', 'red', 'gray']
            ),
            row=2, col=1
        )
        
        # Optimization opportunities
        fig.add_trace(
            go.Scatter(
                x=journey_analysis['acquisition_cost'],
                y=journey_analysis['predicted_clv'],
                mode='markers',
                marker=dict(
                    color=journey_analysis['ltv_cac_ratio'],
                    colorscale='RdYlGn',
                    size=8,
                    opacity=0.6,
                    colorbar=dict(title="LTV/CAC Ratio")
                ),
                name="Customers"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Customer Journey & Lifecycle Analysis",
            height=800,
            showlegend=True
        )
        
        fig.write_html('visualizations/customer_journey_analysis.html')
        print("✅ Customer journey analysis saved to visualizations/customer_journey_analysis.html")
        
        return fig
    
    def generate_all_scenarios(self):
        """Generate all business scenario visualizations"""
        print("🚀 Generating All Business Scenario Visualizations...")
        
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate all scenarios
        self.create_retention_campaign_analysis()
        self.create_revenue_optimization_dashboard()
        self.create_marketing_budget_allocation()
        self.create_customer_journey_analysis()
        
        print("\n✅ All business scenario visualizations completed!")
        print("📁 Files generated:")
        print("   - visualizations/retention_campaign_analysis.html")
        print("   - visualizations/revenue_optimization_dashboard.html")
        print("   - visualizations/marketing_budget_allocation.html")
        print("   - visualizations/customer_journey_analysis.html")
        print("   - data/processed/marketing_budget_recommendations.csv")

if __name__ == "__main__":
    visualizer = BusinessScenarioVisualizer()
    visualizer.generate_all_scenarios()