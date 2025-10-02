"""
Geospatial Analysis for Retail Site Selection
Advanced geospatial analytics for optimal retail location selection
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class GeospatialRetailAnalytics:
    def __init__(self):
        self.location_data = None
        self.site_recommendations = []
        
    def generate_location_data(self, n_locations=500):
        """Generate realistic location dataset with demographics"""
        np.random.seed(42)
        
        # City centers (lat, lon)
        city_centers = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740)
        }
        
        locations = []
        for i in range(n_locations):
            # Select random city
            city = np.random.choice(list(city_centers.keys()))
            city_lat, city_lon = city_centers[city]
            
            # Generate location around city center
            lat_offset = np.random.normal(0, 0.1)  # ~11km radius
            lon_offset = np.random.normal(0, 0.1)
            
            latitude = city_lat + lat_offset
            longitude = city_lon + lon_offset
            
            # Demographics
            population_density = np.random.lognormal(6, 1)  # People per sq km
            median_income = np.random.normal(65000, 20000)
            median_income = max(25000, median_income)
            
            # Age demographics
            age_0_18 = np.random.uniform(0.15, 0.35)
            age_19_35 = np.random.uniform(0.20, 0.40)
            age_36_55 = np.random.uniform(0.25, 0.35)
            age_55_plus = 1 - (age_0_18 + age_19_35 + age_36_55)
            
            # Competition analysis
            num_competitors = np.random.poisson(3)
            distance_to_nearest_competitor = np.random.exponential(2)  # km
            
            # Accessibility metrics
            distance_to_highway = np.random.exponential(5)  # km
            public_transport_score = np.random.uniform(1, 10)
            parking_availability = np.random.uniform(0.3, 1.0)
            
            # Foot traffic estimation
            foot_traffic_weekday = np.random.lognormal(7, 0.5)
            foot_traffic_weekend = foot_traffic_weekday * np.random.uniform(1.2, 2.0)
            
            # Commercial rent (per sq ft per month)
            rent_per_sqft = np.random.lognormal(3, 0.5)
            
            # Calculate location score
            income_score = min(10, median_income / 10000)
            density_score = min(10, population_density / 1000)
            competition_score = max(1, 10 - num_competitors)
            accessibility_score = (public_transport_score + (10 - min(10, distance_to_highway)) + parking_availability * 10) / 3
            traffic_score = min(10, (foot_traffic_weekday + foot_traffic_weekend) / 2000)
            
            overall_score = (income_score + density_score + competition_score + accessibility_score + traffic_score) / 5
            
            locations.append({
                'location_id': f'LOC{i+1:04d}',
                'city': city,
                'latitude': latitude,
                'longitude': longitude,
                'population_density': population_density,
                'median_income': median_income,
                'age_0_18': age_0_18,
                'age_19_35': age_19_35,
                'age_36_55': age_36_55,
                'age_55_plus': age_55_plus,
                'num_competitors': num_competitors,
                'distance_to_nearest_competitor': distance_to_nearest_competitor,
                'distance_to_highway': distance_to_highway,
                'public_transport_score': public_transport_score,
                'parking_availability': parking_availability,
                'foot_traffic_weekday': foot_traffic_weekday,
                'foot_traffic_weekend': foot_traffic_weekend,
                'rent_per_sqft': rent_per_sqft,
                'location_score': overall_score
            })
        
        self.data = pd.DataFrame(locations)
        
        print(f"✅ Generated {len(self.data):,} location records")
        print(f"   - Cities covered: {len(city_centers)}")
        print(f"   - Average location score: {self.data['location_score'].mean():.2f}")
        print(f"   - Average median income: ${self.data['median_income'].mean():,.0f}")
        
        return self.data
    
    def analyze_site_potential(self):
        """Analyze site potential and generate recommendations"""
        print("🔄 Analyzing site potential...")
        
        # Define criteria weights
        weights = {
            'income': 0.25,
            'density': 0.20,
            'competition': 0.15,
            'accessibility': 0.20,
            'traffic': 0.20
        }
        
        # Normalize scores (0-10 scale)
        self.data['income_score'] = np.clip(self.data['median_income'] / 10000, 0, 10)
        self.data['density_score'] = np.clip(self.data['population_density'] / 1000, 0, 10)
        self.data['competition_score'] = np.clip(10 - self.data['num_competitors'], 1, 10)
        self.data['accessibility_score'] = (
            self.data['public_transport_score'] + 
            np.clip(10 - self.data['distance_to_highway'], 0, 10) + 
            self.data['parking_availability'] * 10
        ) / 3
        self.data['traffic_score'] = np.clip((self.data['foot_traffic_weekday'] + self.data['foot_traffic_weekend']) / 2000, 0, 10)
        
        # Calculate weighted score
        self.data['weighted_score'] = (
            self.data['income_score'] * weights['income'] +
            self.data['density_score'] * weights['density'] +
            self.data['competition_score'] * weights['competition'] +
            self.data['accessibility_score'] * weights['accessibility'] +
            self.data['traffic_score'] * weights['traffic']
        ) * 10  # Scale to 0-100
        
        # ROI estimation
        # Assume revenue is correlated with foot traffic and income
        estimated_monthly_revenue = (
            self.data['foot_traffic_weekday'] * 22 + 
            self.data['foot_traffic_weekend'] * 8
        ) * (self.data['median_income'] / 50000) * 5  # $5 average transaction
        
        # Costs (rent + operational)
        monthly_rent = self.data['rent_per_sqft'] * 2000  # Assume 2000 sq ft
        operational_costs = estimated_monthly_revenue * 0.6  # 60% of revenue
        total_monthly_costs = monthly_rent + operational_costs
        
        # Monthly profit and ROI
        monthly_profit = estimated_monthly_revenue - total_monthly_costs
        initial_investment = 200000  # $200k initial investment
        annual_roi = (monthly_profit * 12) / initial_investment
        
        self.data['estimated_monthly_revenue'] = estimated_monthly_revenue
        self.data['monthly_rent'] = monthly_rent
        self.data['monthly_profit'] = monthly_profit
        self.data['annual_roi'] = annual_roi
        
        # Risk assessment
        self.data['risk_score'] = (
            (self.data['num_competitors'] / 10) * 30 +  # Competition risk
            (self.data['rent_per_sqft'] / self.data['rent_per_sqft'].max()) * 30 +  # Cost risk
            (1 - self.data['parking_availability']) * 20 +  # Accessibility risk
            (self.data['distance_to_highway'] / 20) * 20  # Location risk
        )
        
        print("✅ Site potential analysis complete!")
    
    def generate_site_recommendations(self, top_n=10):
        """Generate top site recommendations"""
        print("🔄 Generating site recommendations...")
        
        # Filter viable locations (positive ROI, reasonable risk)
        viable_locations = self.data[
            (self.data['annual_roi'] > 0.1) &  # At least 10% ROI
            (self.data['risk_score'] < 70) &   # Risk score below 70
            (self.data['weighted_score'] > 50)  # Overall score above 50
        ].copy()
        
        # Rank by combined score (weighted score + ROI - risk)
        viable_locations['recommendation_score'] = (
            viable_locations['weighted_score'] * 0.4 +
            viable_locations['annual_roi'] * 100 * 0.4 +
            (100 - viable_locations['risk_score']) * 0.2
        )
        
        # Get top recommendations
        top_recommendations = viable_locations.nlargest(top_n, 'recommendation_score')
        
        self.site_recommendations = []
        for _, location in top_recommendations.iterrows():
            self.site_recommendations.append({
                'location_id': location['location_id'],
                'city': location['city'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'recommendation_score': location['recommendation_score'],
                'weighted_score': location['weighted_score'],
                'annual_roi': location['annual_roi'],
                'risk_score': location['risk_score'],
                'estimated_monthly_revenue': location['estimated_monthly_revenue'],
                'monthly_profit': location['monthly_profit'],
                'key_strengths': self._identify_strengths(location),
                'potential_concerns': self._identify_concerns(location)
            })
        
        print(f"✅ Generated {len(self.site_recommendations)} site recommendations")
    
    def _identify_strengths(self, location):
        """Identify key strengths of a location"""
        strengths = []
        
        if location['income_score'] > 7:
            strengths.append("High-income demographics")
        if location['density_score'] > 7:
            strengths.append("High population density")
        if location['competition_score'] > 7:
            strengths.append("Low competition")
        if location['accessibility_score'] > 7:
            strengths.append("Excellent accessibility")
        if location['traffic_score'] > 7:
            strengths.append("High foot traffic")
        
        return strengths[:3]  # Top 3 strengths
    
    def _identify_concerns(self, location):
        """Identify potential concerns of a location"""
        concerns = []
        
        if location['rent_per_sqft'] > self.data['rent_per_sqft'].quantile(0.8):
            concerns.append("High rental costs")
        if location['num_competitors'] > 5:
            concerns.append("High competition")
        if location['distance_to_highway'] > 10:
            concerns.append("Limited highway access")
        if location['parking_availability'] < 0.5:
            concerns.append("Limited parking")
        
        return concerns[:2]  # Top 2 concerns
    
    def run_complete_analysis(self):
        """Run complete geospatial retail analysis"""
        print("🚀 Starting Geospatial Retail Site Selection Analysis...")
        
        # Generate and analyze data
        self.generate_location_data()
        self.analyze_site_potential()
        self.generate_site_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/location_analysis.csv', index=False)
        
        # Save recommendations
        recommendations_df = pd.DataFrame(self.site_recommendations)
        recommendations_df.to_csv('results/site_recommendations.csv', index=False)
        
        import json
        with open('results/geospatial_analysis.json', 'w') as f:
            json.dump({
                'top_recommendations': self.site_recommendations,
                'analysis_summary': {
                    'total_locations_analyzed': len(self.data),
                    'viable_locations': len(self.data[self.data['annual_roi'] > 0.1]),
                    'average_roi': f"{self.data['annual_roi'].mean():.1%}",
                    'best_city': self.data.groupby('city')['weighted_score'].mean().idxmax()
                }
            }, f, indent=2, default=str)
        
        print("\n✅ Geospatial Retail Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/location_analysis.csv")
        print("   - results/site_recommendations.csv")
        print("   - results/geospatial_analysis.json")
        
        avg_roi = self.data['annual_roi'].mean()
        best_location = self.site_recommendations[0] if self.site_recommendations else None
        
        return {
            'status': 'complete',
            'locations_analyzed': len(self.data),
            'recommendations_generated': len(self.site_recommendations),
            'average_roi': f"{avg_roi:.1%}",
            'best_location_score': f"{best_location['recommendation_score']:.1f}" if best_location else "N/A"
        }

if __name__ == "__main__":
    geo_system = GeospatialRetailAnalytics()
    results = geo_system.run_complete_analysis()