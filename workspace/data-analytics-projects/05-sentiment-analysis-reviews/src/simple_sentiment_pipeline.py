"""
Customer Review Sentiment Analysis System (Simplified)
Basic sentiment analysis with business insights
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings('ignore')

class SimpleSentimentAnalysisSystem:
    def __init__(self):
        # Simple sentiment word lists
        self.positive_words = [
            'excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'perfect', 'outstanding',
            'love', 'best', 'awesome', 'brilliant', 'superb', 'magnificent', 'incredible',
            'good', 'nice', 'happy', 'satisfied', 'pleased', 'recommend', 'quality'
        ]
        
        self.negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
            'poor', 'useless', 'broken', 'defective', 'waste', 'money', 'refund',
            'problem', 'issue', 'wrong', 'damaged', 'cheap', 'overpriced', 'fake'
        ]
        
    def generate_review_data(self, n_reviews=10000):
        """Generate realistic customer review dataset"""
        np.random.seed(42)
        
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        
        positive_templates = [
            "Great product! Highly recommend it. Excellent quality and fast shipping.",
            "Amazing quality! Worth every penny. Will definitely buy again.",
            "Perfect! Exactly what I was looking for. Great customer service too.",
            "Outstanding product! Exceeded my expectations completely.",
            "Fantastic quality and great value for money. Love it!"
        ]
        
        negative_templates = [
            "Terrible product! Poor quality and waste of money.",
            "Awful! Broke after one day. Very disappointed.",
            "Poor quality! Not as described. Requesting refund.",
            "Horrible experience! Product defective and customer service unhelpful.",
            "Terrible! Cheap materials and poor construction."
        ]
        
        neutral_templates = [
            "Okay product. Nothing special but does the job.",
            "Average quality. Could be better for the price.",
            "It's fine. Not great but not terrible either.",
            "Decent product. Meets basic expectations.",
            "Okay. Works as expected but nothing extraordinary."
        ]
        
        reviews = []
        
        for i in range(n_reviews):
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
            
            if sentiment_type == 'positive':
                review_text = np.random.choice(positive_templates)
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif sentiment_type == 'negative':
                review_text = np.random.choice(negative_templates)
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            else:
                review_text = np.random.choice(neutral_templates)
                rating = 3
            
            reviews.append({
                'review_id': f'R{i+1:05d}',
                'product_category': np.random.choice(categories),
                'review_text': review_text,
                'rating': rating,
                'review_date': pd.date_range(start='2023-01-01', end='2024-01-01', periods=n_reviews)[i],
                'verified_purchase': np.random.choice([True, False], p=[0.8, 0.2]),
                'helpful_votes': np.random.poisson(3)
            })
        
        self.data = pd.DataFrame(reviews)
        
        print(f"✅ Generated {len(self.data):,} customer reviews")
        print(f"   - Categories: {len(self.data['product_category'].unique())}")
        print(f"   - Date range: {self.data['review_date'].min()} to {self.data['review_date'].max()}")
        print(f"   - Average rating: {self.data['rating'].mean():.2f}")
        
        return self.data
    
    def simple_sentiment_analysis(self, text):
        """Simple rule-based sentiment analysis"""
        text = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        if positive_count > negative_count:
            return 'Positive', 0.5 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            return 'Negative', -0.5 - (negative_count - positive_count) * 0.1
        else:
            return 'Neutral', 0.0
    
    def perform_sentiment_analysis(self):
        """Perform sentiment analysis on all reviews"""
        print("🔄 Performing sentiment analysis...")
        
        sentiments = []
        scores = []
        
        for text in self.data['review_text']:
            sentiment, score = self.simple_sentiment_analysis(text)
            sentiments.append(sentiment)
            scores.append(score)
        
        self.data['sentiment'] = sentiments
        self.data['sentiment_score'] = scores
        
        print("✅ Sentiment analysis complete!")
        print(f"   - Positive reviews: {(self.data['sentiment'] == 'Positive').sum():,}")
        print(f"   - Negative reviews: {(self.data['sentiment'] == 'Negative').sum():,}")
        print(f"   - Neutral reviews: {(self.data['sentiment'] == 'Neutral').sum():,}")
    
    def create_sentiment_dashboard(self):
        """Create comprehensive sentiment analysis dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Sentiment Distribution', 'Rating vs Sentiment', 'Sentiment Over Time',
                          'Category Sentiment', 'Sentiment Scores', 'Monthly Trends',
                          'Verified vs Unverified', 'Helpful Votes Analysis', 'Category Performance'),
            specs=[[{"type": "pie"}, {"type": "box"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment Distribution
        sentiment_counts = self.data['sentiment'].value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, name="Sentiment"),
            row=1, col=1
        )
        
        # 2. Rating vs Sentiment
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_data = self.data[self.data['sentiment'] == sentiment]
            fig.add_trace(
                go.Box(y=sentiment_data['rating'], name=sentiment),
                row=1, col=2
            )
        
        # 3. Sentiment Over Time
        daily_sentiment = self.data.groupby([self.data['review_date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        for sentiment in daily_sentiment.columns:
            fig.add_trace(
                go.Scatter(x=daily_sentiment.index, y=daily_sentiment[sentiment], 
                          mode='lines', name=f'{sentiment} Trend'),
                row=1, col=3
            )
        
        # 4. Category Sentiment
        category_sentiment = pd.crosstab(self.data['product_category'], self.data['sentiment'], normalize='index') * 100
        for sentiment in category_sentiment.columns:
            fig.add_trace(
                go.Bar(x=category_sentiment.index, y=category_sentiment[sentiment], name=sentiment),
                row=2, col=1
            )
        
        # 5. Sentiment Score Distribution
        fig.add_trace(
            go.Histogram(x=self.data['sentiment_score'], nbinsx=50, name='Sentiment Score'),
            row=2, col=2
        )
        
        # 6. Monthly Review Trends
        monthly_reviews = self.data.groupby(self.data['review_date'].dt.to_period('M')).size()
        fig.add_trace(
            go.Bar(x=[str(x) for x in monthly_reviews.index], y=monthly_reviews.values, name='Monthly Reviews'),
            row=2, col=3
        )
        
        # 7. Verified vs Unverified
        verified_sentiment = pd.crosstab(self.data['verified_purchase'], self.data['sentiment'], normalize='index') * 100
        for sentiment in verified_sentiment.columns:
            fig.add_trace(
                go.Bar(x=['Unverified', 'Verified'], y=verified_sentiment.loc[[False, True], sentiment], 
                      name=f'{sentiment} %'),
                row=3, col=1
            )
        
        # 8. Helpful Votes vs Sentiment
        fig.add_trace(
            go.Scatter(x=self.data['helpful_votes'], y=self.data['sentiment_score'], 
                      mode='markers', opacity=0.6, name='Helpful Votes vs Sentiment'),
            row=3, col=2
        )
        
        # 9. Category Performance
        category_stats = self.data.groupby('product_category').agg({
            'rating': 'mean',
            'sentiment_score': 'mean'
        }).round(2)
        
        fig.add_trace(
            go.Bar(x=category_stats.index, y=category_stats['rating'], name='Avg Rating'),
            row=3, col=3
        )
        
        fig.update_layout(height=1200, title_text="Customer Review Sentiment Analysis Dashboard")
        return fig
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        overall_sentiment = self.data['sentiment'].value_counts(normalize=True) * 100
        category_sentiment = self.data.groupby('product_category')['sentiment_score'].mean().sort_values()
        
        recommendations = {
            'immediate_actions': [
                f"Focus on improving {category_sentiment.index[0]} category (lowest sentiment score)",
                f"Address negative reviews - currently {overall_sentiment.get('Negative', 0):.1f}% of all reviews",
                "Implement sentiment monitoring for real-time customer feedback",
                "Set up automated alerts for sudden drops in sentiment scores"
            ],
            'product_improvements': [
                "Analyze negative reviews for common product issues",
                "Improve product quality based on recurring complaints",
                "Enhance product descriptions to match customer expectations",
                "Consider product improvements for consistently negative items"
            ],
            'customer_service': [
                "Respond proactively to negative reviews within 24 hours",
                "Implement customer satisfaction follow-up surveys",
                "Train customer service team on sentiment-aware responses",
                "Create escalation procedures for highly negative feedback"
            ],
            'marketing_insights': [
                f"Leverage positive sentiment ({overall_sentiment.get('Positive', 0):.1f}%) in marketing campaigns",
                "Use positive review quotes in product descriptions",
                "Target marketing efforts on high-sentiment categories",
                "Create case studies from highly satisfied customers"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete sentiment analysis"""
        print("🚀 Starting Customer Review Sentiment Analysis...")
        
        # Generate and analyze data
        self.generate_review_data()
        self.perform_sentiment_analysis()
        
        # Create visualizations
        dashboard = self.create_sentiment_dashboard()
        
        # Generate recommendations
        recommendations = self.generate_business_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        dashboard.write_html('results/sentiment_analysis_dashboard.html')
        self.data.to_csv('results/sentiment_analysis_data.csv', index=False)
        
        import json
        with open('results/sentiment_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'dataset_summary': {
                'total_reviews': len(self.data),
                'categories': len(self.data['product_category'].unique()),
                'average_rating': f"{self.data['rating'].mean():.2f}"
            },
            'sentiment_analysis': {
                'positive_reviews': f"{(self.data['sentiment'] == 'Positive').sum():,} ({(self.data['sentiment'] == 'Positive').mean()*100:.1f}%)",
                'negative_reviews': f"{(self.data['sentiment'] == 'Negative').sum():,} ({(self.data['sentiment'] == 'Negative').mean()*100:.1f}%)",
                'neutral_reviews': f"{(self.data['sentiment'] == 'Neutral').sum():,} ({(self.data['sentiment'] == 'Neutral').mean()*100:.1f}%)"
            },
            'key_insights': [
                f"Overall sentiment: {(self.data['sentiment'] == 'Positive').mean()*100:.1f}% positive",
                f"Best category: {self.data.groupby('product_category')['sentiment_score'].mean().idxmax()}",
                f"Worst category: {self.data.groupby('product_category')['sentiment_score'].mean().idxmin()}"
            ]
        }
        
        with open('results/sentiment_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("\n✅ Sentiment Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/sentiment_analysis_dashboard.html")
        print("   - results/sentiment_analysis_data.csv")
        print("   - results/sentiment_recommendations.json")
        print("   - results/sentiment_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Total Reviews: {len(self.data):,}")
        print(f"   - Positive Sentiment: {(self.data['sentiment'] == 'Positive').mean()*100:.1f}%")
        print(f"   - Categories Analyzed: {len(self.data['product_category'].unique())}")
        
        return summary_report

if __name__ == "__main__":
    sentiment_system = SimpleSentimentAnalysisSystem()
    results = sentiment_system.run_complete_analysis()