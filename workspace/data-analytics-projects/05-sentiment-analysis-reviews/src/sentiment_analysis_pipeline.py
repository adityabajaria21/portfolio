"""
Customer Review Sentiment Analysis System
NLP-powered sentiment analysis with business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalysisSystem:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def generate_review_data(self, n_reviews=10000):
        """Generate realistic customer review dataset"""
        np.random.seed(42)
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        
        # Positive review templates
        positive_templates = [
            "Great product! Highly recommend it. Excellent quality and fast shipping.",
            "Amazing quality! Worth every penny. Will definitely buy again.",
            "Perfect! Exactly what I was looking for. Great customer service too.",
            "Outstanding product! Exceeded my expectations completely.",
            "Fantastic quality and great value for money. Love it!",
            "Excellent product! Fast delivery and perfect packaging.",
            "Amazing! Better than expected. Highly satisfied with purchase.",
            "Perfect quality! Great design and functionality. Recommended!",
            "Outstanding value! Great product at reasonable price.",
            "Excellent! Top quality product with great customer support."
        ]
        
        # Negative review templates
        negative_templates = [
            "Terrible product! Poor quality and waste of money.",
            "Awful! Broke after one day. Very disappointed.",
            "Poor quality! Not as described. Requesting refund.",
            "Horrible experience! Product defective and customer service unhelpful.",
            "Terrible! Cheap materials and poor construction.",
            "Awful quality! Not worth the money at all.",
            "Poor product! Doesn't work as advertised.",
            "Horrible! Worst purchase ever. Completely useless.",
            "Terrible experience! Product arrived damaged.",
            "Awful! Poor quality and overpriced."
        ]
        
        # Neutral review templates
        neutral_templates = [
            "Okay product. Nothing special but does the job.",
            "Average quality. Could be better for the price.",
            "It's fine. Not great but not terrible either.",
            "Decent product. Meets basic expectations.",
            "Okay. Works as expected but nothing extraordinary.",
            "Average. Good enough for the price point.",
            "Fine product. Does what it's supposed to do.",
            "Decent quality. Could use some improvements.",
            "Okay experience. Product is acceptable.",
            "Average. Nothing to complain about but not impressive."
        ]
        
        reviews = []
        
        for i in range(n_reviews):
            # Determine sentiment distribution: 40% positive, 30% negative, 30% neutral
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                            p=[0.4, 0.3, 0.3])
            
            if sentiment_type == 'positive':
                review_text = np.random.choice(positive_templates)
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif sentiment_type == 'negative':
                review_text = np.random.choice(negative_templates)
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            else:  # neutral
                review_text = np.random.choice(neutral_templates)
                rating = 3
            
            # Add some variation to reviews
            if np.random.random() < 0.3:  # 30% chance to add extra text
                extra_phrases = [
                    " The delivery was quick.",
                    " Packaging was good.",
                    " Customer service was helpful.",
                    " Would recommend to friends.",
                    " Good value for money.",
                    " Easy to use.",
                    " Stylish design.",
                    " Durable material."
                ]
                review_text += np.random.choice(extra_phrases)
            
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
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def perform_sentiment_analysis(self):
        """Perform comprehensive sentiment analysis"""
        print("🔄 Performing sentiment analysis...")
        
        # VADER sentiment analysis
        sentiment_scores = []
        for text in self.data['review_text']:
            scores = self.sia.polarity_scores(text)
            sentiment_scores.append(scores)
        
        # Add sentiment scores to dataframe
        sentiment_df = pd.DataFrame(sentiment_scores)
        self.data = pd.concat([self.data, sentiment_df], axis=1)
        
        # Classify sentiment based on compound score
        def classify_sentiment(compound_score):
            if compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.data['sentiment'] = self.data['compound'].apply(classify_sentiment)
        
        # Preprocess text for ML model
        self.data['processed_text'] = self.data['review_text'].apply(self.preprocess_text)
        
        print("✅ Sentiment analysis complete!")
        print(f"   - Positive reviews: {(self.data['sentiment'] == 'Positive').sum():,}")
        print(f"   - Negative reviews: {(self.data['sentiment'] == 'Negative').sum():,}")
        print(f"   - Neutral reviews: {(self.data['sentiment'] == 'Neutral').sum():,}")
    
    def train_ml_sentiment_model(self):
        """Train machine learning model for sentiment classification"""
        print("🔄 Training ML sentiment model...")
        
        # Prepare data for ML
        X = self.vectorizer.fit_transform(self.data['processed_text'])
        y = self.data['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.ml_model.predict(X_test)
        
        # Store results
        self.ml_results = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_names': self.vectorizer.get_feature_names_out()
        }
        
        accuracy = self.ml_results['classification_report']['accuracy']
        print(f"✅ ML model trained! Accuracy: {accuracy:.3f}")
    
    def create_sentiment_overview_dashboard(self):
        """Create comprehensive sentiment analysis dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Sentiment Distribution', 'Rating vs Sentiment', 'Sentiment Over Time',
                          'Category Sentiment Analysis', 'Sentiment Score Distribution', 'Word Cloud Positive',
                          'Top Positive Words', 'Top Negative Words', 'Sentiment by Category'),
            specs=[[{"type": "pie"}, {"type": "box"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
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
        
        # 4. Category Sentiment Analysis
        category_sentiment = pd.crosstab(self.data['product_category'], self.data['sentiment'], normalize='index') * 100
        for sentiment in category_sentiment.columns:
            fig.add_trace(
                go.Bar(x=category_sentiment.index, y=category_sentiment[sentiment], name=sentiment),
                row=2, col=1
            )
        
        # 5. Sentiment Score Distribution
        fig.add_trace(
            go.Histogram(x=self.data['compound'], nbinsx=50, name='Compound Score'),
            row=2, col=2
        )
        
        # 6. Placeholder for word cloud (will be text-based)
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=['Word Cloud<br>Generated<br>Separately'], 
                      textfont=dict(size=16)),
            row=2, col=3
        )
        
        # 7. Top Positive Words
        positive_reviews = self.data[self.data['sentiment'] == 'Positive']['processed_text']
        positive_text = ' '.join(positive_reviews)
        positive_words = pd.Series(positive_text.split()).value_counts().head(10)
        fig.add_trace(
            go.Bar(x=positive_words.values, y=positive_words.index, orientation='h', name='Positive Words'),
            row=3, col=1
        )
        
        # 8. Top Negative Words
        negative_reviews = self.data[self.data['sentiment'] == 'Negative']['processed_text']
        negative_text = ' '.join(negative_reviews)
        negative_words = pd.Series(negative_text.split()).value_counts().head(10)
        fig.add_trace(
            go.Bar(x=negative_words.values, y=negative_words.index, orientation='h', name='Negative Words'),
            row=3, col=2
        )
        
        # 9. Average Sentiment Score by Category
        avg_sentiment = self.data.groupby('product_category')['compound'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=avg_sentiment.index, y=avg_sentiment.values, name='Avg Sentiment Score'),
            row=3, col=3
        )
        
        fig.update_layout(height=1200, title_text="Customer Review Sentiment Analysis Dashboard")
        return fig
    
    def create_business_insights_dashboard(self):
        """Create business-focused insights dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Sentiment vs Rating Correlation', 'Review Volume Trends', 'Verified vs Unverified',
                          'Helpful Votes Analysis', 'Category Performance', 'Sentiment Prediction Accuracy'),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment vs Rating Correlation
        fig.add_trace(
            go.Scatter(x=self.data['rating'], y=self.data['compound'], 
                      mode='markers', opacity=0.6, name='Rating vs Sentiment'),
            row=1, col=1
        )
        
        # 2. Review Volume Trends
        monthly_reviews = self.data.groupby(self.data['review_date'].dt.to_period('M')).size()
        fig.add_trace(
            go.Bar(x=[str(x) for x in monthly_reviews.index], y=monthly_reviews.values, name='Monthly Reviews'),
            row=1, col=2
        )
        
        # 3. Verified vs Unverified Purchase Sentiment
        verified_sentiment = pd.crosstab(self.data['verified_purchase'], self.data['sentiment'], normalize='index') * 100
        for sentiment in verified_sentiment.columns:
            fig.add_trace(
                go.Bar(x=['Unverified', 'Verified'], y=verified_sentiment.loc[[False, True], sentiment], 
                      name=f'{sentiment} %'),
                row=1, col=3
            )
        
        # 4. Helpful Votes vs Sentiment
        fig.add_trace(
            go.Scatter(x=self.data['helpful_votes'], y=self.data['compound'], 
                      mode='markers', opacity=0.6, name='Helpful Votes vs Sentiment'),
            row=2, col=1
        )
        
        # 5. Category Performance (Average Rating and Sentiment)
        category_stats = self.data.groupby('product_category').agg({
            'rating': 'mean',
            'compound': 'mean'
        }).round(2)
        
        fig.add_trace(
            go.Bar(x=category_stats.index, y=category_stats['rating'], name='Avg Rating'),
            row=2, col=2
        )
        
        # 6. ML Model Performance
        if hasattr(self, 'ml_results'):
            precision_scores = [self.ml_results['classification_report'][sentiment]['precision'] 
                              for sentiment in ['Negative', 'Neutral', 'Positive']]
            fig.add_trace(
                go.Bar(x=['Negative', 'Neutral', 'Positive'], y=precision_scores, name='Precision'),
                row=2, col=3
            )
        
        fig.update_layout(height=800, title_text="Business Intelligence - Sentiment Analysis Insights")
        return fig
    
    def generate_word_clouds(self):
        """Generate word clouds for different sentiments"""
        word_clouds = {}
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_text = ' '.join(self.data[self.data['sentiment'] == sentiment]['processed_text'])
            
            if sentiment_text.strip():  # Check if there's text to process
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    max_words=100,
                                    colormap='viridis' if sentiment == 'Positive' else 'Reds' if sentiment == 'Negative' else 'Blues').generate(sentiment_text)
                word_clouds[sentiment] = wordcloud
        
        return word_clouds
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        # Calculate key metrics
        overall_sentiment = self.data['sentiment'].value_counts(normalize=True) * 100
        category_sentiment = self.data.groupby('product_category')['compound'].mean().sort_values()
        rating_sentiment_corr = self.data['rating'].corr(self.data['compound'])
        
        # Identify problem areas
        negative_categories = category_sentiment[category_sentiment < -0.1].index.tolist()
        low_rated_reviews = self.data[self.data['rating'] <= 2]
        
        recommendations = {
            'immediate_actions': [
                f"Focus on improving {negative_categories[0] if negative_categories else 'lowest-performing'} category (lowest sentiment score)",
                f"Address {len(low_rated_reviews)} low-rated reviews with negative sentiment",
                f"Implement sentiment monitoring for real-time customer feedback",
                "Set up automated alerts for sudden drops in sentiment scores"
            ],
            'product_improvements': [
                "Analyze negative reviews for common product issues",
                "Improve product quality based on recurring complaints",
                "Enhance product descriptions to match customer expectations",
                "Consider product recalls or improvements for consistently negative items"
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
            ],
            'monitoring_metrics': [
                "Daily sentiment score tracking by category",
                "Weekly sentiment trend analysis",
                "Monthly customer satisfaction correlation with sales",
                "Quarterly sentiment impact on business metrics"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete sentiment analysis"""
        print("🚀 Starting Customer Review Sentiment Analysis...")
        
        # Generate and analyze data
        self.generate_review_data()
        self.perform_sentiment_analysis()
        self.train_ml_sentiment_model()
        
        # Create visualizations
        overview_dashboard = self.create_sentiment_overview_dashboard()
        business_dashboard = self.create_business_insights_dashboard()
        
        # Generate word clouds
        word_clouds = self.generate_word_clouds()
        
        # Generate recommendations
        recommendations = self.generate_business_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        overview_dashboard.write_html('results/sentiment_overview_dashboard.html')
        business_dashboard.write_html('results/sentiment_business_dashboard.html')
        
        # Save word clouds
        for sentiment, wordcloud in word_clouds.items():
            wordcloud.to_file(f'results/wordcloud_{sentiment.lower()}.png')
        
        # Save data and recommendations
        self.data.to_csv('results/sentiment_analysis_data.csv', index=False)
        
        import json
        with open('results/sentiment_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'dataset_summary': {
                'total_reviews': len(self.data),
                'date_range': f"{self.data['review_date'].min()} to {self.data['review_date'].max()}",
                'categories': len(self.data['product_category'].unique()),
                'average_rating': f"{self.data['rating'].mean():.2f}"
            },
            'sentiment_analysis': {
                'positive_reviews': f"{(self.data['sentiment'] == 'Positive').sum():,} ({(self.data['sentiment'] == 'Positive').mean()*100:.1f}%)",
                'negative_reviews': f"{(self.data['sentiment'] == 'Negative').sum():,} ({(self.data['sentiment'] == 'Negative').mean()*100:.1f}%)",
                'neutral_reviews': f"{(self.data['sentiment'] == 'Neutral').sum():,} ({(self.data['sentiment'] == 'Neutral').mean()*100:.1f}%)",
                'ml_model_accuracy': f"{self.ml_results['classification_report']['accuracy']:.3f}"
            },
            'key_insights': [
                f"Overall sentiment distribution: {(self.data['sentiment'] == 'Positive').mean()*100:.1f}% positive",
                f"Rating-sentiment correlation: {self.data['rating'].corr(self.data['compound']):.3f}",
                f"Best performing category: {self.data.groupby('product_category')['compound'].mean().idxmax()}",
                f"Worst performing category: {self.data.groupby('product_category')['compound'].mean().idxmin()}"
            ]
        }
        
        with open('results/sentiment_analysis_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print("\n✅ Sentiment Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/sentiment_overview_dashboard.html")
        print("   - results/sentiment_business_dashboard.html")
        print("   - results/wordcloud_positive.png")
        print("   - results/wordcloud_negative.png")
        print("   - results/wordcloud_neutral.png")
        print("   - results/sentiment_analysis_data.csv")
        print("   - results/sentiment_recommendations.json")
        print("   - results/sentiment_analysis_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Total Reviews: {len(self.data):,}")
        print(f"   - Positive Sentiment: {(self.data['sentiment'] == 'Positive').mean()*100:.1f}%")
        print(f"   - ML Model Accuracy: {self.ml_results['classification_report']['accuracy']:.3f}")
        print(f"   - Categories Analyzed: {len(self.data['product_category'].unique())}")
        
        return summary_report

if __name__ == "__main__":
    sentiment_system = SentimentAnalysisSystem()
    results = sentiment_system.run_complete_analysis()