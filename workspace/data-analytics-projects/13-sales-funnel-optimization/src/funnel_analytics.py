"""
Sales Funnel & Conversion Rate Optimization System
Comprehensive funnel analysis with conversion optimization
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class SalesFunnelAnalytics:
    def __init__(self):
        self.funnel_data = None
        
    def generate_funnel_data(self, n_users=100000):
        """Generate realistic sales funnel dataset"""
        np.random.seed(42)
        
        # Define funnel stages
        stages = [
            'Landing Page View',
            'Product Page View', 
            'Add to Cart',
            'Checkout Started',
            'Payment Info',
            'Purchase Complete'
        ]
        
        # Base conversion rates for each stage
        base_conversion_rates = [1.0, 0.6, 0.3, 0.8, 0.9, 0.85]
        
        users = []
        for i in range(n_users):
            user_id = f'U{i+1:06d}'
            
            # User characteristics that affect conversion
            device_type = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.5, 0.4, 0.1])
            traffic_source = np.random.choice(['Organic', 'Paid', 'Direct', 'Social'], p=[0.4, 0.3, 0.2, 0.1])
            user_segment = np.random.choice(['New', 'Returning'], p=[0.7, 0.3])
            
            # Adjust conversion rates based on characteristics
            device_multiplier = {'Desktop': 1.0, 'Mobile': 0.8, 'Tablet': 0.9}[device_type]
            source_multiplier = {'Organic': 1.0, 'Paid': 1.2, 'Direct': 1.1, 'Social': 0.9}[traffic_source]
            segment_multiplier = {'New': 1.0, 'Returning': 1.3}[user_segment]
            
            total_multiplier = device_multiplier * source_multiplier * segment_multiplier
            
            # Simulate user journey through funnel
            current_stage = 0
            session_value = np.random.uniform(50, 500)  # Potential order value
            
            for stage_idx, stage in enumerate(stages):
                if stage_idx == 0:
                    # Everyone starts at landing page
                    reached_stage = True
                else:
                    # Calculate probability of reaching this stage
                    conversion_rate = base_conversion_rates[stage_idx] * total_multiplier
                    conversion_rate = min(1.0, max(0.0, conversion_rate))  # Clamp between 0 and 1
                    reached_stage = np.random.random() < conversion_rate
                
                if reached_stage:
                    current_stage = stage_idx
                    
                    users.append({
                        'user_id': user_id,
                        'session_date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_users)[i],
                        'device_type': device_type,
                        'traffic_source': traffic_source,
                        'user_segment': user_segment,
                        'stage': stage,
                        'stage_order': stage_idx,
                        'session_value': session_value,
                        'completed': 1
                    })
                else:
                    # User dropped off at this stage
                    break
        
        self.data = pd.DataFrame(users)
        
        print(f"✅ Generated {len(self.data):,} funnel events for {n_users:,} users")
        print(f"   - Stages: {len(stages)}")
        print(f"   - Date range: {self.data['session_date'].min()} to {self.data['session_date'].max()}")
        
        return self.data
    
    def analyze_funnel_performance(self):
        """Analyze funnel conversion rates and drop-offs"""
        print("🔄 Analyzing funnel performance...")
        
        # Overall funnel analysis
        funnel_summary = self.data.groupby('stage').agg({
            'user_id': 'nunique',
            'session_value': 'mean'
        }).reset_index()
        
        # Add stage order for proper sorting
        stage_order = {stage: idx for idx, stage in enumerate(self.data['stage'].unique())}
        funnel_summary['stage_order'] = funnel_summary['stage'].map(stage_order)
        funnel_summary = funnel_summary.sort_values('stage_order')
        
        # Calculate conversion rates
        total_users = funnel_summary.iloc[0]['user_id']
        funnel_summary['conversion_rate'] = funnel_summary['user_id'] / total_users
        funnel_summary['drop_off_rate'] = 1 - funnel_summary['conversion_rate']
        
        # Stage-to-stage conversion
        funnel_summary['stage_conversion'] = funnel_summary['user_id'] / funnel_summary['user_id'].shift(1)
        funnel_summary['stage_conversion'].iloc[0] = 1.0  # First stage is 100%
        
        self.funnel_summary = funnel_summary
        
        # Segment analysis
        self.segment_analysis = {}
        
        for segment_col in ['device_type', 'traffic_source', 'user_segment']:
            segment_funnel = self.data.groupby([segment_col, 'stage'])['user_id'].nunique().reset_index()
            segment_pivot = segment_funnel.pivot(index=segment_col, columns='stage', values='user_id')
            
            # Calculate conversion rates for each segment
            segment_conversion = segment_pivot.div(segment_pivot.iloc[:, 0], axis=0)
            self.segment_analysis[segment_col] = segment_conversion
        
        print("✅ Funnel analysis complete!")
        print(f"   - Overall conversion rate: {funnel_summary.iloc[-1]['conversion_rate']:.2%}")
        print(f"   - Biggest drop-off: {funnel_summary.loc[funnel_summary['stage_conversion'].idxmin(), 'stage']}")
    
    def identify_optimization_opportunities(self):
        """Identify areas for funnel optimization"""
        print("🔄 Identifying optimization opportunities...")
        
        # Find stages with highest drop-off rates
        stage_drop_offs = []
        for i in range(1, len(self.funnel_summary)):
            current_stage = self.funnel_summary.iloc[i]
            prev_stage = self.funnel_summary.iloc[i-1]
            
            drop_off_users = prev_stage['user_id'] - current_stage['user_id']
            drop_off_rate = drop_off_users / prev_stage['user_id']
            
            stage_drop_offs.append({
                'from_stage': prev_stage['stage'],
                'to_stage': current_stage['stage'],
                'users_lost': drop_off_users,
                'drop_off_rate': drop_off_rate,
                'potential_revenue_lost': drop_off_users * current_stage['session_value']
            })
        
        self.optimization_opportunities = pd.DataFrame(stage_drop_offs)
        self.optimization_opportunities = self.optimization_opportunities.sort_values('drop_off_rate', ascending=False)
        
        # Segment-specific opportunities
        self.segment_opportunities = {}
        
        for segment_col in ['device_type', 'traffic_source', 'user_segment']:
            segment_data = self.segment_analysis[segment_col]
            
            # Find segments with lowest conversion rates
            final_stage = segment_data.columns[-1]
            segment_performance = segment_data[final_stage].sort_values()
            
            self.segment_opportunities[segment_col] = {
                'best_performing': segment_performance.idxmax(),
                'worst_performing': segment_performance.idxmin(),
                'performance_gap': segment_performance.max() - segment_performance.min()
            }
        
        print("✅ Optimization analysis complete!")
    
    def run_complete_analysis(self):
        """Run complete sales funnel analysis"""
        print("🚀 Starting Sales Funnel Analysis...")
        
        # Generate and analyze data
        self.generate_funnel_data()
        self.analyze_funnel_performance()
        self.identify_optimization_opportunities()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/funnel_data.csv', index=False)
        self.funnel_summary.to_csv('results/funnel_summary.csv', index=False)
        self.optimization_opportunities.to_csv('results/optimization_opportunities.csv', index=False)
        
        import json
        with open('results/segment_analysis.json', 'w') as f:
            json.dump({
                'segment_opportunities': self.segment_opportunities
            }, f, indent=2, default=str)
        
        print("\n✅ Sales Funnel Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/funnel_data.csv")
        print("   - results/funnel_summary.csv")
        print("   - results/optimization_opportunities.csv")
        print("   - results/segment_analysis.json")
        
        overall_conversion = self.funnel_summary.iloc[-1]['conversion_rate']
        biggest_opportunity = self.optimization_opportunities.iloc[0]
        
        return {
            'status': 'complete',
            'overall_conversion_rate': f"{overall_conversion:.2%}",
            'biggest_drop_off_stage': biggest_opportunity['from_stage'],
            'potential_revenue_opportunity': f"${biggest_opportunity['potential_revenue_lost']:,.2f}"
        }

if __name__ == "__main__":
    funnel_system = SalesFunnelAnalytics()
    results = funnel_system.run_complete_analysis()