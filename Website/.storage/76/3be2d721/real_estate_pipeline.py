"""
Real Estate Price Prediction System
ML-powered property valuation with market analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RealEstatePredictionSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def generate_real_estate_data(self, n_properties=5000):
        """Generate realistic real estate dataset"""
        np.random.seed(42)
        
        # Location data
        neighborhoods = ['Downtown', 'Suburbs', 'Waterfront', 'Historic', 'Business District', 
                        'Residential', 'Industrial', 'University Area', 'Shopping District', 'Park View']
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Luxury']
        
        data = []
        
        for i in range(n_properties):
            # Basic property features
            bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
            bathrooms = max(1, bedrooms - np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
            
            # Square footage based on bedrooms
            base_sqft = bedrooms * 400 + np.random.normal(200, 100)
            square_feet = max(500, int(base_sqft))
            
            # Lot size
            lot_size = np.random.lognormal(8, 0.5)  # Square feet
            
            # Property age
            age = np.random.exponential(15)  # Years
            age = min(age, 100)  # Cap at 100 years
            
            # Location factors
            city = np.random.choice(cities)
            neighborhood = np.random.choice(neighborhoods)
            
            # Distance to amenities (miles)
            distance_to_downtown = np.random.exponential(5)
            distance_to_school = np.random.exponential(2)
            distance_to_transit = np.random.exponential(1)
            
            # Property type
            property_type = np.random.choice(property_types, 
                                           p=[0.4, 0.25, 0.15, 0.1, 0.1])
            
            # Additional features
            garage_spaces = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.4, 0.1])
            has_pool = np.random.choice([0, 1], p=[0.8, 0.2])
            has_fireplace = np.random.choice([0, 1], p=[0.6, 0.4])
            has_basement = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Market conditions
            market_score = np.random.normal(0.5, 0.2)  # 0-1 scale
            market_score = max(0, min(1, market_score))
            
            # Calculate base price using realistic factors
            base_price = 100000  # Base price
            
            # City multiplier
            city_multipliers = {
                'New York': 3.5, 'Los Angeles': 2.8, 'San Francisco': 3.2,
                'Chicago': 1.5, 'Houston': 1.2, 'Phoenix': 1.3,
                'Philadelphia': 1.4, 'San Antonio': 1.0, 'San Diego': 2.5,
                'Dallas': 1.3, 'San Jose': 3.0
            }
            
            price = base_price * city_multipliers.get(city, 1.5)
            
            # Square footage impact
            price += square_feet * np.random.normal(150, 20)
            
            # Bedroom/bathroom impact
            price += bedrooms * 15000 + bathrooms * 8000
            
            # Age depreciation
            price *= (1 - age * 0.005)  # 0.5% depreciation per year
            
            # Neighborhood premium/discount
            neighborhood_multipliers = {
                'Downtown': 1.3, 'Waterfront': 1.5, 'Historic': 1.2,
                'Business District': 1.1, 'Park View': 1.4,
                'Suburbs': 1.0, 'Residential': 0.95, 'Industrial': 0.8,
                'University Area': 1.1, 'Shopping District': 1.05
            }
            price *= neighborhood_multipliers.get(neighborhood, 1.0)
            
            # Property type premium
            type_multipliers = {
                'Single Family': 1.0, 'Condo': 0.85, 'Townhouse': 0.9,
                'Multi-Family': 1.2, 'Luxury': 2.0
            }
            price *= type_multipliers.get(property_type, 1.0)
            
            # Additional features
            price += garage_spaces * 5000
            price += has_pool * 25000
            price += has_fireplace * 8000
            price += has_basement * 15000
            
            # Distance penalties
            price -= distance_to_downtown * 2000
            price -= distance_to_school * 3000
            price -= distance_to_transit * 1500
            
            # Market conditions
            price *= (0.8 + market_score * 0.4)  # 80% to 120% based on market
            
            # Add some noise
            price *= np.random.normal(1, 0.1)
            price = max(50000, int(price))  # Minimum price floor
            
            data.append({
                'property_id': f'P{i+1:05d}',
                'city': city,
                'neighborhood': neighborhood,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'square_feet': square_feet,
                'lot_size': lot_size,
                'age': age,
                'garage_spaces': garage_spaces,
                'has_pool': has_pool,
                'has_fireplace': has_fireplace,
                'has_basement': has_basement,
                'distance_to_downtown': distance_to_downtown,
                'distance_to_school': distance_to_school,
                'distance_to_transit': distance_to_transit,
                'market_score': market_score,
                'price': price
            })
        
        self.data = pd.DataFrame(data)
        
        # Add derived features
        self.data['price_per_sqft'] = self.data['price'] / self.data['square_feet']
        self.data['room_ratio'] = self.data['bathrooms'] / self.data['bedrooms']
        self.data['total_rooms'] = self.data['bedrooms'] + self.data['bathrooms']
        self.data['is_luxury'] = (self.data['property_type'] == 'Luxury').astype(int)
        self.data['amenity_score'] = (self.data['has_pool'] + self.data['has_fireplace'] + 
                                     self.data['has_basement'] + self.data['garage_spaces']/3)
        
        print(f"✅ Generated {len(self.data):,} property records")
        print(f"   - Cities: {len(self.data['city'].unique())}")
        print(f"   - Neighborhoods: {len(self.data['neighborhood'].unique())}")
        print(f"   - Price range: ${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}")
        print(f"   - Average price: ${self.data['price'].mean():,.0f}")
        
        return self.data
    
    def prepare_data_for_modeling(self):
        """Prepare data for machine learning models"""
        # Encode categorical variables
        categorical_cols = ['city', 'neighborhood', 'property_type']
        
        df_encoded = self.data.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
            self.encoders[col] = le
        
        # Feature columns for modeling
        feature_cols = ['bedrooms', 'bathrooms', 'square_feet', 'lot_size', 'age',
                       'garage_spaces', 'has_pool', 'has_fireplace', 'has_basement',
                       'distance_to_downtown', 'distance_to_school', 'distance_to_transit',
                       'market_score', 'room_ratio', 'total_rooms', 'amenity_score'] + \
                      [col + '_encoded' for col in categorical_cols]
        
        X = df_encoded[feature_cols]
        y = df_encoded['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train, X_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = feature_cols
        self.scaler = scaler
        
        print(f"✅ Data prepared for modeling")
        print(f"   - Features: {len(feature_cols)}")
        print(f"   - Training samples: {len(X_train):,}")
        print(f"   - Test samples: {len(X_test):,}")
    
    def train_price_prediction_models(self):
        """Train multiple price prediction models"""
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.model_results = {}
        
        for name, model in models_config.items():
            print(f"🔄 Training {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'Linear' in name or 'Ridge' in name:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test_use)
            
            # Metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            self.models[name] = model
            self.model_results[name] = {
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2
            }
            
            print(f"   ✅ {name} - R²: {r2:.4f}, MAE: ${mae:,.0f}")
    
    def create_real_estate_overview_dashboard(self):
        """Create comprehensive real estate analysis dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Price Distribution', 'Price by City', 'Price vs Square Feet',
                          'Price by Property Type', 'Age vs Price', 'Neighborhood Analysis',
                          'Amenities Impact', 'Market Score Analysis', 'Price per Sq Ft by City'),
            specs=[[{"type": "histogram"}, {"type": "box"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Price Distribution
        fig.add_trace(
            go.Histogram(x=self.data['price'], nbinsx=50, name='Price Distribution'),
            row=1, col=1
        )
        
        # 2. Price by City
        for city in self.data['city'].unique()[:5]:  # Top 5 cities
            city_data = self.data[self.data['city'] == city]
            fig.add_trace(
                go.Box(y=city_data['price'], name=city),
                row=1, col=2
            )
        
        # 3. Price vs Square Feet
        fig.add_trace(
            go.Scatter(x=self.data['square_feet'], y=self.data['price'], 
                      mode='markers', opacity=0.6, name='Price vs Sq Ft'),
            row=1, col=3
        )
        
        # 4. Price by Property Type
        type_prices = self.data.groupby('property_type')['price'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=type_prices.index, y=type_prices.values, name='Avg Price by Type'),
            row=2, col=1
        )
        
        # 5. Age vs Price
        fig.add_trace(
            go.Scatter(x=self.data['age'], y=self.data['price'], 
                      mode='markers', opacity=0.6, name='Age vs Price'),
            row=2, col=2
        )
        
        # 6. Neighborhood Analysis
        neighborhood_prices = self.data.groupby('neighborhood')['price'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=neighborhood_prices.index, y=neighborhood_prices.values, name='Avg Price by Neighborhood'),
            row=2, col=3
        )
        
        # 7. Amenities Impact
        amenity_impact = {
            'Pool': self.data.groupby('has_pool')['price'].mean().diff().iloc[-1],
            'Fireplace': self.data.groupby('has_fireplace')['price'].mean().diff().iloc[-1],
            'Basement': self.data.groupby('has_basement')['price'].mean().diff().iloc[-1]
        }
        fig.add_trace(
            go.Bar(x=list(amenity_impact.keys()), y=list(amenity_impact.values()), name='Amenity Premium'),
            row=3, col=1
        )
        
        # 8. Market Score Analysis
        fig.add_trace(
            go.Scatter(x=self.data['market_score'], y=self.data['price'], 
                      mode='markers', opacity=0.6, name='Market Score vs Price'),
            row=3, col=2
        )
        
        # 9. Price per Sq Ft by City
        price_per_sqft_city = self.data.groupby('city')['price_per_sqft'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=price_per_sqft_city.index, y=price_per_sqft_city.values, name='Price per Sq Ft'),
            row=3, col=3
        )
        
        fig.update_layout(height=1200, title_text="Real Estate Market Analysis Dashboard")
        return fig
    
    def create_model_performance_dashboard(self):
        """Create model performance comparison dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Prediction vs Actual', 
                          'Residual Analysis', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Model Performance Comparison
        model_names = list(self.model_results.keys())
        r2_scores = [self.model_results[name]['r2_score'] for name in model_names]
        mae_scores = [self.model_results[name]['mae'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R² Score'),
            row=1, col=1
        )
        
        # 2. Prediction vs Actual (Best Model)
        best_model_name = max(model_names, key=lambda x: self.model_results[x]['r2_score'])
        best_predictions = self.model_results[best_model_name]['predictions']
        
        fig.add_trace(
            go.Scatter(x=self.y_test, y=best_predictions, mode='markers', 
                      name=f'{best_model_name} Predictions'),
            row=1, col=2
        )
        
        # Add perfect prediction line
        min_price = min(self.y_test.min(), best_predictions.min())
        max_price = max(self.y_test.max(), best_predictions.max())
        fig.add_trace(
            go.Scatter(x=[min_price, max_price], y=[min_price, max_price], 
                      mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
            row=1, col=2
        )
        
        # 3. Residual Analysis
        residuals = self.y_test - best_predictions
        fig.add_trace(
            go.Scatter(x=best_predictions, y=residuals, mode='markers', name='Residuals'),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importance = rf_model.feature_importances_
            
            # Get top 10 features
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], 
                       orientation='h', name='Feature Importance'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Price Prediction Model Performance Analysis")
        return fig
    
    def create_investment_analysis_dashboard(self):
        """Create investment opportunity analysis dashboard"""
        # Calculate investment metrics
        self.data['value_score'] = (self.data['price_per_sqft'] - self.data['price_per_sqft'].mean()) / self.data['price_per_sqft'].std()
        self.data['market_potential'] = self.data['market_score'] * (1 - self.data['age']/100)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Investment Opportunities', 'Value vs Market Potential', 'ROI by Property Type',
                          'Price Trends by Neighborhood', 'Risk Assessment', 'Investment Recommendations'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Investment Opportunities (Price vs Value Score)
        fig.add_trace(
            go.Scatter(x=self.data['price'], y=self.data['value_score'], 
                      mode='markers', opacity=0.6, name='Investment Opportunities'),
            row=1, col=1
        )
        
        # 2. Value vs Market Potential
        fig.add_trace(
            go.Scatter(x=self.data['value_score'], y=self.data['market_potential'], 
                      mode='markers', opacity=0.6, name='Value vs Potential'),
            row=1, col=2
        )
        
        # 3. ROI by Property Type
        roi_by_type = self.data.groupby('property_type')['market_potential'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=roi_by_type.index, y=roi_by_type.values, name='Market Potential by Type'),
            row=1, col=3
        )
        
        # 4. Price Trends by Neighborhood
        neighborhood_stats = self.data.groupby('neighborhood').agg({
            'price': 'mean',
            'market_potential': 'mean'
        }).sort_values('market_potential', ascending=False)
        
        fig.add_trace(
            go.Bar(x=neighborhood_stats.index, y=neighborhood_stats['price'], name='Avg Price'),
            row=2, col=1
        )
        
        # 5. Risk Assessment (Age vs Price)
        fig.add_trace(
            go.Scatter(x=self.data['age'], y=self.data['price'], 
                      mode='markers', opacity=0.6, name='Age vs Price Risk'),
            row=2, col=2
        )
        
        # 6. Investment Recommendations Table
        # Find top investment opportunities
        investment_opportunities = self.data[
            (self.data['value_score'] < -0.5) & (self.data['market_potential'] > 0.6)
        ].nlargest(5, 'market_potential')[['property_id', 'city', 'neighborhood', 'price', 'market_potential']]
        
        if len(investment_opportunities) > 0:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Property ID', 'City', 'Neighborhood', 'Price', 'Market Potential']),
                    cells=dict(values=[
                        investment_opportunities['property_id'],
                        investment_opportunities['city'],
                        investment_opportunities['neighborhood'],
                        [f"${x:,.0f}" for x in investment_opportunities['price']],
                        [f"{x:.2f}" for x in investment_opportunities['market_potential']]
                    ])
                ),
                row=2, col=3
            )
        
        fig.update_layout(height=800, title_text="Real Estate Investment Analysis Dashboard")
        return fig
    
    def generate_market_insights(self):
        """Generate comprehensive market insights and recommendations"""
        # Market analysis
        avg_price_by_city = self.data.groupby('city')['price'].mean().sort_values(ascending=False)
        best_value_neighborhoods = self.data.groupby('neighborhood')['price_per_sqft'].mean().sort_values()
        
        # Investment opportunities
        undervalued_properties = self.data[
            (self.data['value_score'] < -1) & (self.data['market_potential'] > 0.5)
        ]
        
        # Model performance
        best_model = max(self.model_results.keys(), key=lambda x: self.model_results[x]['r2_score'])
        
        insights = {
            'market_overview': {
                'total_properties_analyzed': len(self.data),
                'average_price': f"${self.data['price'].mean():,.0f}",
                'price_range': f"${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}",
                'most_expensive_city': avg_price_by_city.index[0],
                'most_affordable_city': avg_price_by_city.index[-1]
            },
            'investment_opportunities': [
                f"Found {len(undervalued_properties)} undervalued properties with high market potential",
                f"Best value neighborhood: {best_value_neighborhoods.index[0]} (${best_value_neighborhoods.iloc[0]:.0f}/sqft)",
                f"Luxury properties show {((self.data[self.data['property_type']=='Luxury']['price'].mean() / self.data['price'].mean() - 1) * 100):.0f}% premium",
                "Properties with pools command average $25,000 premium"
            ],
            'market_trends': [
                f"Newer properties (< 10 years) average ${self.data[self.data['age']<10]['price'].mean():,.0f}",
                f"Properties near downtown (< 2 miles) show {((self.data[self.data['distance_to_downtown']<2]['price'].mean() / self.data['price'].mean() - 1) * 100):.0f}% premium",
                f"High market score areas (> 0.7) have {((self.data[self.data['market_score']>0.7]['price'].mean() / self.data['price'].mean() - 1) * 100):.0f}% higher prices",
                "Single family homes represent the largest market segment"
            ],
            'pricing_insights': [
                f"Average price per square foot: ${self.data['price_per_sqft'].mean():.0f}",
                f"Properties with 4+ bedrooms command {((self.data[self.data['bedrooms']>=4]['price'].mean() / self.data[self.data['bedrooms']<4]['price'].mean() - 1) * 100):.0f}% premium",
                f"Garage spaces add average ${(self.data.groupby('garage_spaces')['price'].mean().diff().mean()):,.0f} per space",
                "Waterfront properties show highest neighborhood premium"
            ],
            'model_performance': {
                'best_model': best_model,
                'r2_score': f"{self.model_results[best_model]['r2_score']:.4f}",
                'mae': f"${self.model_results[best_model]['mae']:,.0f}",
                'prediction_accuracy': f"{(1 - self.model_results[best_model]['mae'] / self.data['price'].mean()) * 100:.1f}%"
            }
        }
        
        return insights
    
    def run_complete_analysis(self):
        """Run complete real estate analysis"""
        print("🚀 Starting Real Estate Price Prediction Analysis...")
        
        # Generate and prepare data
        self.generate_real_estate_data()
        self.prepare_data_for_modeling()
        
        # Train models
        self.train_price_prediction_models()
        
        # Create visualizations
        overview_dashboard = self.create_real_estate_overview_dashboard()
        model_dashboard = self.create_model_performance_dashboard()
        investment_dashboard = self.create_investment_analysis_dashboard()
        
        # Generate insights
        market_insights = self.generate_market_insights()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        overview_dashboard.write_html('results/real_estate_overview_dashboard.html')
        model_dashboard.write_html('results/price_prediction_model_dashboard.html')
        investment_dashboard.write_html('results/investment_analysis_dashboard.html')
        
        # Save data and insights
        self.data.to_csv('results/real_estate_data.csv', index=False)
        
        import json
        with open('results/market_insights.json', 'w') as f:
            json.dump(market_insights, f, indent=2)
        
        # Generate summary report
        best_model = max(self.model_results.keys(), key=lambda x: self.model_results[x]['r2_score'])
        
        summary_report = {
            'dataset_summary': {
                'total_properties': len(self.data),
                'cities_covered': len(self.data['city'].unique()),
                'neighborhoods': len(self.data['neighborhood'].unique()),
                'price_range': f"${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}"
            },
            'model_performance': {
                'best_model': best_model,
                'r2_score': f"{self.model_results[best_model]['r2_score']:.4f}",
                'mae': f"${self.model_results[best_model]['mae']:,.0f}",
                'rmse': f"${self.model_results[best_model]['rmse']:,.0f}"
            },
            'market_insights': {
                'average_price': f"${self.data['price'].mean():,.0f}",
                'average_price_per_sqft': f"${self.data['price_per_sqft'].mean():.0f}",
                'most_expensive_city': self.data.groupby('city')['price'].mean().idxmax(),
                'best_value_neighborhood': self.data.groupby('neighborhood')['price_per_sqft'].mean().idxmin()
            }
        }
        
        with open('results/real_estate_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("\n✅ Real Estate Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/real_estate_overview_dashboard.html")
        print("   - results/price_prediction_model_dashboard.html")
        print("   - results/investment_analysis_dashboard.html")
        print("   - results/real_estate_data.csv")
        print("   - results/market_insights.json")
        print("   - results/real_estate_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Properties Analyzed: {len(self.data):,}")
        print(f"   - Best Model: {best_model}")
        print(f"   - Model R² Score: {self.model_results[best_model]['r2_score']:.4f}")
        print(f"   - Average Price: ${self.data['price'].mean():,.0f}")
        print(f"   - Prediction MAE: ${self.model_results[best_model]['mae']:,.0f}")
        
        return summary_report

if __name__ == "__main__":
    real_estate_system = RealEstatePredictionSystem()
    results = real_estate_system.run_complete_analysis()