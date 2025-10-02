"""
Gaming Platform User Behavior & Monetization Analytics
Advanced gaming analytics for player behavior and revenue optimization
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class GamingAnalytics:
    def __init__(self):
        self.player_data = None
        self.monetization_insights = {}
        
    def generate_gaming_data(self, n_players=100000):
        """Generate realistic gaming dataset"""
        np.random.seed(42)
        
        players = []
        for i in range(n_players):
            # Player demographics
            age = np.random.choice([13, 16, 18, 25, 30, 35, 40], p=[0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.1])
            gender = np.random.choice(['M', 'F', 'Other'], p=[0.6, 0.35, 0.05])
            country = np.random.choice(['US', 'UK', 'DE', 'JP', 'KR', 'CN', 'BR'], p=[0.3, 0.1, 0.08, 0.12, 0.1, 0.2, 0.1])
            
            # Gaming behavior
            registration_date = pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_players)[i]
            
            # Player type affects behavior
            player_type = np.random.choice(['Casual', 'Core', 'Hardcore'], p=[0.6, 0.3, 0.1])
            
            if player_type == 'Casual':
                avg_session_duration = np.random.normal(15, 5)  # minutes
                sessions_per_week = np.random.poisson(3)
                spending_propensity = 0.1
            elif player_type == 'Core':
                avg_session_duration = np.random.normal(45, 15)
                sessions_per_week = np.random.poisson(8)
                spending_propensity = 0.3
            else:  # Hardcore
                avg_session_duration = np.random.normal(120, 30)
                sessions_per_week = np.random.poisson(15)
                spending_propensity = 0.6
            
            avg_session_duration = max(5, avg_session_duration)
            sessions_per_week = max(0, sessions_per_week)
            
            # Calculate total playtime and engagement
            weeks_since_registration = min(52, (pd.Timestamp('2024-01-01') - registration_date).days / 7)
            total_sessions = int(sessions_per_week * weeks_since_registration)
            total_playtime = total_sessions * avg_session_duration
            
            # Level progression (based on playtime)
            level = min(100, int(total_playtime / 60) + 1)  # 1 hour per level
            
            # Monetization
            is_paying_user = np.random.random() < spending_propensity
            if is_paying_user:
                # Spending varies by player type and country
                country_multiplier = {'US': 1.0, 'UK': 0.8, 'DE': 0.9, 'JP': 1.2, 'KR': 1.1, 'CN': 0.6, 'BR': 0.4}[country]
                base_spending = {'Casual': 20, 'Core': 80, 'Hardcore': 200}[player_type]
                total_spent = np.random.exponential(base_spending * country_multiplier)
                
                # Number of purchases
                avg_purchase_amount = np.random.uniform(5, 50)
                num_purchases = max(1, int(total_spent / avg_purchase_amount))
            else:
                total_spent = 0
                num_purchases = 0
            
            # Retention calculation
            days_since_registration = (pd.Timestamp('2024-01-01') - registration_date).days
            
            # Churn probability increases over time
            churn_base_rate = {'Casual': 0.8, 'Core': 0.4, 'Hardcore': 0.2}[player_type]
            churn_prob = churn_base_rate * (1 - np.exp(-days_since_registration / 90))  # 90-day half-life
            
            is_churned = np.random.random() < churn_prob
            
            if is_churned:
                # Last login was some time ago
                days_since_last_login = np.random.randint(8, days_since_registration + 1)
            else:
                days_since_last_login = np.random.randint(0, 7)  # Active in last week
            
            # Social features
            friends_count = np.random.poisson({'Casual': 2, 'Core': 8, 'Hardcore': 15}[player_type])
            guild_member = np.random.random() < {'Casual': 0.2, 'Core': 0.6, 'Hardcore': 0.9}[player_type]
            
            # Performance metrics
            win_rate = np.random.beta(2, 2) * 0.8 + 0.1  # Between 0.1 and 0.9
            avg_score = np.random.lognormal(8, 1)
            
            players.append({
                'player_id': f'P{i+1:06d}',
                'age': age,
                'gender': gender,
                'country': country,
                'player_type': player_type,
                'registration_date': registration_date,
                'level': level,
                'total_sessions': total_sessions,
                'total_playtime': total_playtime,
                'avg_session_duration': avg_session_duration,
                'sessions_per_week': sessions_per_week,
                'is_paying_user': is_paying_user,
                'total_spent': total_spent,
                'num_purchases': num_purchases,
                'days_since_last_login': days_since_last_login,
                'is_churned': is_churned,
                'friends_count': friends_count,
                'guild_member': guild_member,
                'win_rate': win_rate,
                'avg_score': avg_score
            })
        
        self.data = pd.DataFrame(players)
        
        print(f"✅ Generated {len(self.data):,} player records")
        print(f"   - Paying users: {self.data['is_paying_user'].sum():,} ({self.data['is_paying_user'].mean():.1%})")
        print(f"   - Total revenue: ${self.data['total_spent'].sum():,.2f}")
        print(f"   - Churned players: {self.data['is_churned'].sum():,} ({self.data['is_churned'].mean():.1%})")
        
        return self.data
    
    def analyze_player_behavior(self):
        """Analyze player behavior patterns"""
        print("🔄 Analyzing player behavior...")
        
        # Segment analysis
        self.behavior_analysis = {}
        
        # By player type
        type_analysis = self.data.groupby('player_type').agg({
            'total_playtime': 'mean',
            'avg_session_duration': 'mean',
            'sessions_per_week': 'mean',
            'total_spent': 'mean',
            'is_churned': 'mean',
            'level': 'mean',
            'win_rate': 'mean'
        }).round(2)
        
        self.behavior_analysis['by_player_type'] = type_analysis.to_dict()
        
        # By country
        country_analysis = self.data.groupby('country').agg({
            'total_spent': ['mean', 'sum'],
            'is_paying_user': 'mean',
            'is_churned': 'mean',
            'avg_session_duration': 'mean'
        }).round(2)
        
        self.behavior_analysis['by_country'] = country_analysis.to_dict()
        
        # Retention analysis
        self.data['days_since_registration'] = (pd.Timestamp('2024-01-01') - self.data['registration_date']).dt.days
        
        # Day 1, 7, 30 retention
        day_1_retention = (self.data['days_since_last_login'] <= 1).mean()
        day_7_retention = (self.data['days_since_last_login'] <= 7).mean()
        day_30_retention = (self.data['days_since_last_login'] <= 30).mean()
        
        self.behavior_analysis['retention'] = {
            'day_1': day_1_retention,
            'day_7': day_7_retention,
            'day_30': day_30_retention
        }
        
        print("✅ Player behavior analysis complete!")
    
    def analyze_monetization(self):
        """Analyze monetization patterns and opportunities"""
        print("🔄 Analyzing monetization...")
        
        # Revenue metrics
        total_revenue = self.data['total_spent'].sum()
        paying_users = self.data['is_paying_user'].sum()
        conversion_rate = paying_users / len(self.data)
        
        # ARPU and ARPPU
        arpu = total_revenue / len(self.data)  # Average Revenue Per User
        arppu = self.data[self.data['is_paying_user']]['total_spent'].mean()  # Average Revenue Per Paying User
        
        # LTV estimation (simplified)
        avg_lifetime_days = self.data['days_since_registration'].mean()
        avg_daily_revenue = arpu / (avg_lifetime_days / 365)
        estimated_ltv = avg_daily_revenue * 365  # 1-year LTV
        
        # Monetization by segments
        monetization_by_type = self.data.groupby('player_type').agg({
            'total_spent': ['sum', 'mean'],
            'is_paying_user': 'mean'
        }).round(2)
        
        # High-value player identification
        high_value_threshold = self.data['total_spent'].quantile(0.9)
        high_value_players = self.data[self.data['total_spent'] >= high_value_threshold]
        
        # Churn risk among paying users
        paying_churn_rate = self.data[self.data['is_paying_user']]['is_churned'].mean()
        
        self.monetization_insights = {
            'revenue_metrics': {
                'total_revenue': total_revenue,
                'paying_users': paying_users,
                'conversion_rate': conversion_rate,
                'arpu': arpu,
                'arppu': arppu,
                'estimated_ltv': estimated_ltv
            },
            'segment_performance': monetization_by_type.to_dict(),
            'high_value_players': {
                'count': len(high_value_players),
                'threshold': high_value_threshold,
                'avg_spending': high_value_players['total_spent'].mean(),
                'characteristics': {
                    'avg_level': high_value_players['level'].mean(),
                    'avg_playtime': high_value_players['total_playtime'].mean(),
                    'guild_membership_rate': high_value_players['guild_member'].mean()
                }
            },
            'churn_analysis': {
                'overall_churn_rate': self.data['is_churned'].mean(),
                'paying_user_churn_rate': paying_churn_rate,
                'revenue_at_risk': self.data[self.data['is_paying_user'] & self.data['is_churned']]['total_spent'].sum()
            }
        }
        
        print("✅ Monetization analysis complete!")
    
    def generate_optimization_recommendations(self):
        """Generate actionable recommendations for player retention and monetization"""
        print("🔄 Generating optimization recommendations...")
        
        # Analyze patterns for recommendations
        casual_conversion = self.data[self.data['player_type'] == 'Casual']['is_paying_user'].mean()
        hardcore_churn = self.data[self.data['player_type'] == 'Hardcore']['is_churned'].mean()
        
        recommendations = {
            'retention_strategies': [
                f"Focus on Casual player retention - {casual_conversion:.1%} conversion rate shows potential",
                f"Implement early engagement hooks for new players in first 7 days",
                f"Create social features to increase friend connections (current avg: {self.data['friends_count'].mean():.1f})",
                f"Develop guild recruitment campaigns - guild members have {(1-self.data[self.data['guild_member']]['is_churned'].mean()):.1%} retention"
            ],
            'monetization_opportunities': [
                f"Increase Casual player monetization - only {casual_conversion:.1%} currently paying",
                f"Create premium content for Hardcore players (low {hardcore_churn:.1%} churn rate)",
                f"Implement battle pass system for consistent revenue",
                f"Develop country-specific pricing for emerging markets"
            ],
            'product_improvements': [
                f"Optimize session length for Casual players (current: {self.data[self.data['player_type']=='Casual']['avg_session_duration'].mean():.1f} min)",
                f"Add social features to increase friend connections",
                f"Implement skill-based matchmaking to improve win rates",
                f"Create progression systems for different player types"
            ],
            'high_value_player_retention': [
                f"VIP program for top {len(self.data[self.data['total_spent'] >= self.monetization_insights['high_value_players']['threshold']]):,} spenders",
                f"Exclusive content and early access features",
                f"Personal account management for whales",
                f"Community events and tournaments"
            ]
        }
        
        self.optimization_recommendations = recommendations
        
        print("✅ Optimization recommendations generated!")
    
    def run_complete_analysis(self):
        """Run complete gaming analytics"""
        print("🚀 Starting Gaming Platform Analytics...")
        
        # Generate and analyze data
        self.generate_gaming_data()
        self.analyze_player_behavior()
        self.analyze_monetization()
        self.generate_optimization_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/player_data.csv', index=False)
        
        import json
        with open('results/gaming_analytics.json', 'w') as f:
            json.dump({
                'behavior_analysis': self.behavior_analysis,
                'monetization_insights': self.monetization_insights,
                'optimization_recommendations': self.optimization_recommendations
            }, f, indent=2, default=str)
        
        print("\n✅ Gaming Analytics Complete!")
        print("📁 Results saved:")
        print("   - results/player_data.csv")
        print("   - results/gaming_analytics.json")
        
        total_revenue = self.monetization_insights['revenue_metrics']['total_revenue']
        conversion_rate = self.monetization_insights['revenue_metrics']['conversion_rate']
        churn_rate = self.monetization_insights['churn_analysis']['overall_churn_rate']
        
        return {
            'status': 'complete',
            'players_analyzed': len(self.data),
            'total_revenue': f"${total_revenue:,.2f}",
            'conversion_rate': f"{conversion_rate:.1%}",
            'churn_rate': f"{churn_rate:.1%}"
        }

if __name__ == "__main__":
    gaming_system = GamingAnalytics()
    results = gaming_system.run_complete_analysis()