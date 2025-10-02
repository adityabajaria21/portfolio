"""
Dynamic Pricing Optimization System
AI-powered pricing strategy with demand elasticity analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DynamicPricingSystem:
    def __init__(self):
        self.pricing_models = {}
        self.elasticity_data = {}
        
    def generate_pricing_data(self, n_products=50, n_days=365):
        """Generate realistic pricing and sales dataset"""
        np.random.seed(42)
        
        # Product categories with different price sensitivities
        categories = {
            'Electronics': {'base_price': 500, 'elasticity': -1.2, 'seasonality': 0.3},
            'Clothing': {'base_price': 80, 'elasticity': -1.8, 'seasonality': 0.5},
            'Books': {'base_price': 25, 'elasticity': -0.8, 'seasonality': 0.1},
            'Home': {'base_price': 150, 'elasticity': -1.0, 'seasonality': 0.2},
            'Sports': {'base_price': 120, 'elasticity': -1.5, 'seasonality': 0.4}
        }
        
        products = []
        for i in range(n_products):
            category = np.random.choice(list(categories.keys()))
            products.append({
                'product_id': f'P{i+1:03d}',
                'product_name': f'{category} Product {i+1}',
                'category': category,
                'base_price': categories[category]['base_price'] * np.random.uniform(0.7, 1.3),
                'price_elasticity': categories[category]['elasticity'] * np.random.uniform(0.8, 1.2),
                'seasonality_factor': categories[category]['seasonality']
            })
        
        # Generate daily pricing and sales data
        date_range = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        sales_data = []
        for product in products:
            for date in date_range:
                # Seasonal adjustment
                day_of_year = date.timetuple().tm_yday
                seasonal_multiplier = 1 + product['seasonality_factor'] * np.sin(2 * np.pi * day_of_year / 365)
                
                # Competitor pricing influence
                competitor_factor = np.random.uniform(0.9, 1.1)
                
                # Market conditions
                market_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 30)  # Monthly cycles
                
                # Dynamic pricing strategy
                base_price = product['base_price']
                
                # Price variations (±20%)
                price_variation = np.random.uniform(-0.2, 0.2)
                current_price = base_price * (1 + price_variation) * competitor_factor
                
                # Demand calculation based on price elasticity
                price_ratio = current_price / base_price
                demand_multiplier = price_ratio ** product['price_elasticity']
                
                # Base demand with seasonal and market adjustments
                base_demand = 100 * seasonal_multiplier * market_factor
                actual_demand = base_demand * demand_multiplier
                
                # Add noise and ensure positive demand
                actual_demand = max(0, actual_demand + np.random.normal(0, 10))
                
                # Sales (assuming we can meet demand)
                units_sold = int(actual_demand)
                revenue = units_sold * current_price
                
                sales_data.append({
                    'date': date,
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'price': current_price,
                    'base_price': base_price,
                    'units_sold': units_sold,
                    'revenue': revenue,
                    'competitor_price': current_price * np.random.uniform(0.95, 1.05),
                    'inventory_level': np.random.randint(50, 500),
                    'marketing_spend': np.random.uniform(100, 1000),
                    'price_elasticity': product['price_elasticity'],
                    'seasonal_multiplier': seasonal_multiplier,
                    'market_factor': market_factor
                })
        
        self.data = pd.DataFrame(sales_data)
        
        print(f"✅ Generated {len(self.data):,} pricing records")
        print(f"   - Products: {n_products}")
        print(f"   - Days: {n_days}")
        print(f"   - Categories: {len(categories)}")
        print(f"   - Total revenue: ${self.data['revenue'].sum():,.2f}")
        print(f"   - Average price: ${self.data['price'].mean():.2f}")
        
        return self.data
    
    def analyze_price_elasticity(self):
        """Analyze price elasticity for each product"""
        print("🔄 Analyzing price elasticity...")
        
        self.elasticity_results = {}
        
        for product_id in self.data['product_id'].unique():
            product_data = self.data[self.data['product_id'] == product_id].copy()
            
            # Calculate price changes and demand changes
            product_data = product_data.sort_values('date')
            product_data['price_change'] = product_data['price'].pct_change()
            product_data['demand_change'] = product_data['units_sold'].pct_change()
            
            # Remove outliers and NaN values
            clean_data = product_data.dropna()
            clean_data = clean_data[
                (abs(clean_data['price_change']) < 0.5) & 
                (abs(clean_data['demand_change']) < 2.0)
            ]
            
            if len(clean_data) > 10:
                # Calculate elasticity (% change in demand / % change in price)
                elasticity_values = clean_data['demand_change'] / clean_data['price_change']
                elasticity_values = elasticity_values[np.isfinite(elasticity_values)]
                
                if len(elasticity_values) > 0:
                    avg_elasticity = np.median(elasticity_values)  # Use median to reduce outlier impact
                    
                    # Price optimization
                    current_avg_price = product_data['price'].mean()
                    current_avg_demand = product_data['units_sold'].mean()
                    
                    # Optimal price calculation (simplified)
                    if avg_elasticity < -1:  # Elastic demand
                        optimal_price_multiplier = 0.95  # Lower price to increase volume
                    elif avg_elasticity > -0.5:  # Inelastic demand
                        optimal_price_multiplier = 1.05  # Higher price to increase margin
                    else:
                        optimal_price_multiplier = 1.0  # Keep current price
                    
                    optimal_price = current_avg_price * optimal_price_multiplier
                    
                    self.elasticity_results[product_id] = {
                        'product_name': product_data['product_name'].iloc[0],
                        'category': product_data['category'].iloc[0],
                        'current_price': current_avg_price,
                        'optimal_price': optimal_price,
                        'price_elasticity': avg_elasticity,
                        'current_demand': current_avg_demand,
                        'predicted_demand': current_avg_demand * (optimal_price_multiplier ** avg_elasticity),
                        'revenue_impact': (optimal_price * current_avg_demand * (optimal_price_multiplier ** avg_elasticity)) - (current_avg_price * current_avg_demand)
                    }
        
        print(f"✅ Elasticity analysis complete for {len(self.elasticity_results)} products")
        avg_elasticity = np.mean([r['price_elasticity'] for r in self.elasticity_results.values()])
        print(f"   - Average price elasticity: {avg_elasticity:.2f}")
    
    def optimize_pricing_strategy(self):
        """Develop optimized pricing strategies"""
        print("🔄 Optimizing pricing strategies...")
        
        self.pricing_strategies = {}
        
        for category in self.data['category'].unique():
            category_data = self.data[self.data['category'] == category]
            
            # Analyze category performance
            avg_price = category_data['price'].mean()
            avg_revenue = category_data['revenue'].mean()
            price_variance = category_data['price'].std()
            
            # Get elasticity data for category
            category_elasticities = [
                result['price_elasticity'] for product_id, result in self.elasticity_results.items()
                if result['category'] == category
            ]
            
            if category_elasticities:
                avg_elasticity = np.mean(category_elasticities)
                
                # Strategy recommendations
                if avg_elasticity < -1.5:  # Highly elastic
                    strategy = "Competitive Pricing"
                    recommendation = "Focus on volume through competitive pricing"
                    price_adjustment = -0.05  # 5% price reduction
                elif avg_elasticity > -0.8:  # Inelastic
                    strategy = "Premium Pricing"
                    recommendation = "Increase margins through premium pricing"
                    price_adjustment = 0.08  # 8% price increase
                else:  # Moderately elastic
                    strategy = "Value-Based Pricing"
                    recommendation = "Balance price and volume optimization"
                    price_adjustment = 0.02  # 2% price increase
                
                self.pricing_strategies[category] = {
                    'current_avg_price': avg_price,
                    'price_elasticity': avg_elasticity,
                    'strategy': strategy,
                    'recommendation': recommendation,
                    'suggested_price_adjustment': price_adjustment,
                    'expected_revenue_impact': avg_revenue * price_adjustment * (1 + avg_elasticity * price_adjustment)
                }
        
        print(f"✅ Pricing strategies developed for {len(self.pricing_strategies)} categories")
    
    def create_pricing_dashboard(self):
        """Create comprehensive pricing optimization dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Price vs Demand Relationship', 'Revenue by Category', 'Price Elasticity Distribution',
                          'Optimal vs Current Pricing', 'Seasonal Price Patterns', 'Competitor Price Comparison',
                          'Revenue Impact Forecast', 'Price Optimization Opportunities', 'Strategy Recommendations'),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Price vs Demand Relationship
        sample_data = self.data.sample(n=min(1000, len(self.data)))  # Sample for performance
        fig.add_trace(
            go.Scatter(x=sample_data['price'], y=sample_data['units_sold'],
                      mode='markers', opacity=0.6, name='Price vs Demand'),
            row=1, col=1
        )
        
        # 2. Revenue by Category
        category_revenue = self.data.groupby('category')['revenue'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=category_revenue.index, y=category_revenue.values, name='Category Revenue'),
            row=1, col=2
        )
        
        # 3. Price Elasticity Distribution
        if self.elasticity_results:
            elasticities = [r['price_elasticity'] for r in self.elasticity_results.values()]
            fig.add_trace(
                go.Histogram(x=elasticities, nbinsx=20, name='Price Elasticity'),
                row=1, col=3
            )
        
        # 4. Optimal vs Current Pricing
        if self.elasticity_results:
            current_prices = [r['current_price'] for r in self.elasticity_results.values()]
            optimal_prices = [r['optimal_price'] for r in self.elasticity_results.values()]
            
            fig.add_trace(
                go.Scatter(x=current_prices, y=optimal_prices, mode='markers',
                          name='Optimal vs Current Price'),
                row=2, col=1
            )
            
            # Add diagonal line for reference
            min_price = min(min(current_prices), min(optimal_prices))
            max_price = max(max(current_prices), max(optimal_prices))
            fig.add_trace(
                go.Scatter(x=[min_price, max_price], y=[min_price, max_price],
                          mode='lines', line=dict(dash='dash'), name='No Change Line'),
                row=2, col=1
            )
        
        # 5. Seasonal Price Patterns
        monthly_avg_price = self.data.groupby(self.data['date'].dt.month)['price'].mean()
        fig.add_trace(
            go.Scatter(x=monthly_avg_price.index, y=monthly_avg_price.values,
                      mode='lines+markers', name='Monthly Avg Price'),
            row=2, col=2
        )
        
        # 6. Competitor Price Comparison
        fig.add_trace(
            go.Scatter(x=sample_data['price'], y=sample_data['competitor_price'],
                      mode='markers', opacity=0.6, name='Our Price vs Competitor'),
            row=2, col=3
        )
        
        # 7. Revenue Impact Forecast
        if self.elasticity_results:
            revenue_impacts = [r['revenue_impact'] for r in self.elasticity_results.values()]
            product_names = [r['product_name'][:10] + '...' for r in self.elasticity_results.values()]
            
            # Top 10 revenue impact opportunities
            impact_df = pd.DataFrame({
                'product': product_names,
                'impact': revenue_impacts
            }).sort_values('impact', ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(x=impact_df['impact'], y=impact_df['product'],
                       orientation='h', name='Revenue Impact'),
                row=3, col=1
            )
        
        # 8. Price Optimization Opportunities
        if self.elasticity_results:
            # Products with highest optimization potential
            optimization_potential = []
            for result in self.elasticity_results.values():
                potential = abs(result['optimal_price'] - result['current_price']) / result['current_price']
                optimization_potential.append(potential)
            
            fig.add_trace(
                go.Scatter(x=optimization_potential, y=revenue_impacts,
                          mode='markers', name='Optimization Potential'),
                row=3, col=2
            )
        
        # 9. Strategy Recommendations Table
        if hasattr(self, 'pricing_strategies'):
            strategy_df = pd.DataFrame({
                'Category': list(self.pricing_strategies.keys()),
                'Strategy': [s['strategy'] for s in self.pricing_strategies.values()],
                'Price Adjustment': [f"{s['suggested_price_adjustment']:+.1%}" for s in self.pricing_strategies.values()],
                'Expected Impact': [f"${s['expected_revenue_impact']:,.0f}" for s in self.pricing_strategies.values()]
            })
            
            fig.add_trace(
                go.Table(
                    header=dict(values=list(strategy_df.columns)),
                    cells=dict(values=[strategy_df[col] for col in strategy_df.columns])
                ),
                row=3, col=3
            )
        
        fig.update_layout(height=1200, title_text="Dynamic Pricing Optimization Dashboard")
        return fig
    
    def generate_pricing_recommendations(self):
        """Generate actionable pricing recommendations"""
        # Calculate key metrics
        total_revenue_impact = sum(r['revenue_impact'] for r in self.elasticity_results.values())
        avg_elasticity = np.mean([r['price_elasticity'] for r in self.elasticity_results.values()])
        
        # Find best opportunities
        best_opportunities = sorted(
            self.elasticity_results.items(),
            key=lambda x: x[1]['revenue_impact'],
            reverse=True
        )[:5]
        
        recommendations = {
            'executive_summary': {
                'total_products_analyzed': len(self.elasticity_results),
                'potential_revenue_impact': f"${total_revenue_impact:,.2f}",
                'average_price_elasticity': f"{avg_elasticity:.2f}",
                'categories_analyzed': len(self.pricing_strategies) if hasattr(self, 'pricing_strategies') else 0
            },
            'immediate_actions': [
                f"Implement pricing changes for top 5 products with ${sum(r[1]['revenue_impact'] for r in best_opportunities):,.0f} potential impact",
                "Set up automated competitor price monitoring",
                "Implement dynamic pricing for elastic products (elasticity < -1.5)",
                "Review and adjust pricing for inelastic products (elasticity > -0.8)"
            ],
            'pricing_strategies': [
                "Use competitive pricing for highly elastic products",
                "Implement premium pricing for inelastic products",
                "Consider bundle pricing for complementary products",
                "Implement seasonal pricing adjustments based on demand patterns"
            ],
            'technology_recommendations': [
                "Deploy real-time pricing optimization algorithms",
                "Implement A/B testing framework for price changes",
                "Set up automated alerts for competitor price changes",
                "Create customer segment-based pricing models"
            ],
            'monitoring_metrics': [
                "Daily price elasticity tracking",
                "Weekly revenue impact analysis",
                "Monthly competitor price comparison",
                "Quarterly pricing strategy effectiveness review"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete pricing optimization analysis"""
        print("🚀 Starting Dynamic Pricing Optimization Analysis...")
        
        # Generate and analyze data
        self.generate_pricing_data()
        self.analyze_price_elasticity()
        self.optimize_pricing_strategy()
        
        # Create visualizations
        dashboard = self.create_pricing_dashboard()
        
        # Generate recommendations
        recommendations = self.generate_pricing_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        dashboard.write_html('results/pricing_optimization_dashboard.html')
        self.data.to_csv('results/pricing_sales_data.csv', index=False)
        
        # Save elasticity results
        elasticity_df = pd.DataFrame(self.elasticity_results).T
        elasticity_df.to_csv('results/price_elasticity_analysis.csv')
        
        # Save pricing strategies
        if hasattr(self, 'pricing_strategies'):
            strategies_df = pd.DataFrame(self.pricing_strategies).T
            strategies_df.to_csv('results/pricing_strategies.csv')
        
        import json
        with open('results/pricing_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'dataset_summary': {
                'products_analyzed': self.data['product_id'].nunique(),
                'total_records': len(self.data),
                'total_revenue': f"${self.data['revenue'].sum():,.2f}",
                'analysis_period': f"{self.data['date'].min()} to {self.data['date'].max()}"
            },
            'pricing_analysis': {
                'products_with_elasticity_data': len(self.elasticity_results),
                'average_price_elasticity': f"{np.mean([r['price_elasticity'] for r in self.elasticity_results.values()]):.2f}",
                'potential_revenue_impact': f"${sum(r['revenue_impact'] for r in self.elasticity_results.values()):,.2f}",
                'pricing_strategies_developed': len(self.pricing_strategies) if hasattr(self, 'pricing_strategies') else 0
            },
            'key_insights': [
                f"Average price elasticity: {np.mean([r['price_elasticity'] for r in self.elasticity_results.values()]):.2f}",
                f"Total revenue optimization potential: ${sum(r['revenue_impact'] for r in self.elasticity_results.values()):,.0f}",
                f"Most elastic category: {min(self.pricing_strategies.items(), key=lambda x: x[1]['price_elasticity'])[0] if hasattr(self, 'pricing_strategies') else 'N/A'}",
                f"Best revenue opportunity: {max(self.elasticity_results.items(), key=lambda x: x[1]['revenue_impact'])[1]['product_name']}"
            ]
        }
        
        with open('results/pricing_optimization_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print("\n✅ Pricing Optimization Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/pricing_optimization_dashboard.html")
        print("   - results/pricing_sales_data.csv")
        print("   - results/price_elasticity_analysis.csv")
        print("   - results/pricing_strategies.csv")
        print("   - results/pricing_recommendations.json")
        print("   - results/pricing_optimization_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Products Analyzed: {self.data['product_id'].nunique()}")
        print(f"   - Revenue Impact Potential: ${sum(r['revenue_impact'] for r in self.elasticity_results.values()):,.0f}")
        print(f"   - Average Price Elasticity: {np.mean([r['price_elasticity'] for r in self.elasticity_results.values()]):.2f}")
        print(f"   - Pricing Strategies: {len(self.pricing_strategies) if hasattr(self, 'pricing_strategies') else 0}")
        
        return summary_report

if __name__ == "__main__":
    pricing_system = DynamicPricingSystem()
    results = pricing_system.run_complete_analysis()