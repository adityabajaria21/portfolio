"""
Marketing Campaign ROI & Attribution Analysis System
Multi-touch attribution modeling for marketing optimization
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class MarketingAttributionSystem:
    def __init__(self):
        self.attribution_model = None
        
    def generate_marketing_data(self, n_campaigns=200, n_customers=50000):
        """Generate realistic marketing campaign dataset"""
        np.random.seed(42)
        
        # Campaign types and channels
        channels = ['Email', 'Social Media', 'Google Ads', 'Display', 'Direct Mail', 'TV', 'Radio']
        campaign_types = ['Awareness', 'Consideration', 'Conversion', 'Retention']
        
        campaigns = []
        for i in range(n_campaigns):
            channel = np.random.choice(channels)
            campaign_type = np.random.choice(campaign_types)
            
            # Campaign costs vary by channel
            cost_multipliers = {
                'Email': 0.1, 'Social Media': 0.3, 'Google Ads': 1.0,
                'Display': 0.8, 'Direct Mail': 0.6, 'TV': 3.0, 'Radio': 2.0
            }
            
            base_cost = np.random.uniform(5000, 50000)
            campaign_cost = base_cost * cost_multipliers[channel]
            
            # Campaign performance varies by type and channel
            if campaign_type == 'Conversion' and channel in ['Google Ads', 'Email']:
                conversion_rate = np.random.uniform(0.05, 0.15)
            elif campaign_type == 'Awareness' and channel in ['TV', 'Radio']:
                conversion_rate = np.random.uniform(0.01, 0.03)
            else:
                conversion_rate = np.random.uniform(0.02, 0.08)
            
            # Calculate impressions and clicks
            impressions = int(np.random.uniform(10000, 500000))
            ctr = np.random.uniform(0.01, 0.05)  # Click-through rate
            clicks = int(impressions * ctr)
            conversions = int(clicks * conversion_rate)
            
            # Revenue per conversion varies
            avg_order_value = np.random.uniform(50, 500)
            revenue = conversions * avg_order_value
            
            campaigns.append({
                'campaign_id': f'C{i+1:03d}',
                'campaign_name': f'{channel}_{campaign_type}_{i+1}',
                'channel': channel,
                'campaign_type': campaign_type,
                'start_date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_campaigns)[i],
                'cost': campaign_cost,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': revenue,
                'ctr': ctr,
                'conversion_rate': conversion_rate,
                'cpc': campaign_cost / clicks if clicks > 0 else 0,
                'roas': revenue / campaign_cost if campaign_cost > 0 else 0
            })
        
        self.campaign_data = pd.DataFrame(campaigns)
        
        # Generate customer journey data
        customer_journeys = []
        for i in range(n_customers):
            # Number of touchpoints before conversion
            num_touchpoints = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            # Select campaigns for this customer journey
            journey_campaigns = np.random.choice(self.campaign_data['campaign_id'], 
                                               size=num_touchpoints, replace=False)
            
            # Determine if customer converted
            conversion_prob = 0.1 * num_touchpoints  # More touchpoints = higher conversion
            converted = np.random.random() < conversion_prob
            
            if converted:
                conversion_value = np.random.uniform(50, 500)
            else:
                conversion_value = 0
            
            for j, campaign_id in enumerate(journey_campaigns):
                customer_journeys.append({
                    'customer_id': f'CUST{i+1:06d}',
                    'campaign_id': campaign_id,
                    'touchpoint_order': j + 1,
                    'total_touchpoints': num_touchpoints,
                    'converted': int(converted),
                    'conversion_value': conversion_value,
                    'touchpoint_date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_customers)[i] + pd.Timedelta(days=j)
                })
        
        self.journey_data = pd.DataFrame(customer_journeys)
        
        print(f"✅ Generated marketing data:")
        print(f"   - Campaigns: {len(self.campaign_data)}")
        print(f"   - Customer journeys: {len(self.journey_data)}")
        print(f"   - Total marketing spend: ${self.campaign_data['cost'].sum():,.2f}")
        print(f"   - Total revenue: ${self.campaign_data['revenue'].sum():,.2f}")
        
        return self.campaign_data, self.journey_data
    
    def calculate_attribution_models(self):
        """Calculate different attribution models"""
        print("🔄 Calculating attribution models...")
        
        # First-touch attribution
        first_touch = self.journey_data[self.journey_data['touchpoint_order'] == 1]
        first_touch_attribution = first_touch.groupby('campaign_id')['conversion_value'].sum()
        
        # Last-touch attribution
        last_touch = self.journey_data.groupby('customer_id').last().reset_index()
        last_touch_attribution = last_touch.groupby('campaign_id')['conversion_value'].sum()
        
        # Linear attribution (equal credit to all touchpoints)
        linear_attribution = self.journey_data.copy()
        linear_attribution['attributed_value'] = (
            linear_attribution['conversion_value'] / linear_attribution['total_touchpoints']
        )
        linear_attribution_summary = linear_attribution.groupby('campaign_id')['attributed_value'].sum()
        
        # Time-decay attribution (more recent touchpoints get more credit)
        time_decay = self.journey_data.copy()
        time_decay['decay_weight'] = 0.5 ** (time_decay['total_touchpoints'] - time_decay['touchpoint_order'])
        
        # Normalize weights per customer
        weight_sums = time_decay.groupby('customer_id')['decay_weight'].sum()
        time_decay = time_decay.merge(weight_sums.rename('total_weight'), on='customer_id')
        time_decay['normalized_weight'] = time_decay['decay_weight'] / time_decay['total_weight']
        time_decay['attributed_value'] = time_decay['conversion_value'] * time_decay['normalized_weight']
        
        time_decay_attribution = time_decay.groupby('campaign_id')['attributed_value'].sum()
        
        self.attribution_results = {
            'first_touch': first_touch_attribution.to_dict(),
            'last_touch': last_touch_attribution.to_dict(),
            'linear': linear_attribution_summary.to_dict(),
            'time_decay': time_decay_attribution.to_dict()
        }
        
        print("✅ Attribution models calculated!")
    
    def calculate_campaign_roi(self):
        """Calculate ROI for each campaign using different attribution models"""
        print("🔄 Calculating campaign ROI...")
        
        roi_analysis = []
        
        for _, campaign in self.campaign_data.iterrows():
            campaign_id = campaign['campaign_id']
            cost = campaign['cost']
            
            # Get attributed revenue from different models
            first_touch_revenue = self.attribution_results['first_touch'].get(campaign_id, 0)
            last_touch_revenue = self.attribution_results['last_touch'].get(campaign_id, 0)
            linear_revenue = self.attribution_results['linear'].get(campaign_id, 0)
            time_decay_revenue = self.attribution_results['time_decay'].get(campaign_id, 0)
            
            roi_analysis.append({
                'campaign_id': campaign_id,
                'campaign_name': campaign['campaign_name'],
                'channel': campaign['channel'],
                'campaign_type': campaign['campaign_type'],
                'cost': cost,
                'first_touch_revenue': first_touch_revenue,
                'last_touch_revenue': last_touch_revenue,
                'linear_revenue': linear_revenue,
                'time_decay_revenue': time_decay_revenue,
                'first_touch_roi': (first_touch_revenue - cost) / cost if cost > 0 else 0,
                'last_touch_roi': (last_touch_revenue - cost) / cost if cost > 0 else 0,
                'linear_roi': (linear_revenue - cost) / cost if cost > 0 else 0,
                'time_decay_roi': (time_decay_revenue - cost) / cost if cost > 0 else 0
            })
        
        self.roi_analysis = pd.DataFrame(roi_analysis)
        
        print("✅ ROI analysis complete!")
    
    def run_complete_analysis(self):
        """Run complete marketing attribution analysis"""
        print("🚀 Starting Marketing Attribution Analysis...")
        
        # Generate and analyze data
        self.generate_marketing_data()
        self.calculate_attribution_models()
        self.calculate_campaign_roi()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.campaign_data.to_csv('results/campaign_data.csv', index=False)
        self.journey_data.to_csv('results/customer_journey_data.csv', index=False)
        self.roi_analysis.to_csv('results/campaign_roi_analysis.csv', index=False)
        
        import json
        with open('results/attribution_results.json', 'w') as f:
            json.dump(self.attribution_results, f, indent=2, default=str)
        
        print("\n✅ Marketing Attribution Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/campaign_data.csv")
        print("   - results/customer_journey_data.csv")
        print("   - results/campaign_roi_analysis.csv")
        print("   - results/attribution_results.json")
        
        total_spend = self.campaign_data['cost'].sum()
        total_revenue = sum(self.attribution_results['linear'].values())
        overall_roi = (total_revenue - total_spend) / total_spend if total_spend > 0 else 0
        
        return {
            'status': 'complete',
            'campaigns': len(self.campaign_data),
            'total_spend': f"${total_spend:,.2f}",
            'total_revenue': f"${total_revenue:,.2f}",
            'overall_roi': f"{overall_roi:.1%}"
        }

if __name__ == "__main__":
    marketing_system = MarketingAttributionSystem()
    results = marketing_system.run_complete_analysis()