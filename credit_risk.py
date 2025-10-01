"""
Credit Card Fraud Detection Analysis
Advanced machine learning approach with business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionAnalyzer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load or generate credit card transaction data"""
        # Generate synthetic fraud detection dataset
        np.random.seed(42)
        n_samples = 50000
        
        # Normal transactions (99% of data)
        normal_samples = int(n_samples * 0.99)
        normal_data = {
            'Amount': np.random.lognormal(3, 1.5, normal_samples),
            'Time': np.random.uniform(0, 172800, normal_samples),  # 48 hours in seconds
            'V1': np.random.normal(0, 1, normal_samples),
            'V2': np.random.normal(0, 1, normal_samples),
            'V3': np.random.normal(0, 1, normal_samples),
            'V4': np.random.normal(0, 1, normal_samples),
            'V5': np.random.normal(0, 1, normal_samples),
            'Class': np.zeros(normal_samples)
        }
        
        # Fraudulent transactions (1% of data)
        fraud_samples = n_samples - normal_samples
        fraud_data = {
            'Amount': np.random.lognormal(4, 2, fraud_samples),  # Higher amounts
            'Time': np.random.uniform(0, 172800, fraud_samples),
            'V1': np.random.normal(2, 1.5, fraud_samples),  # Different patterns
            'V2': np.random.normal(-1, 1.2, fraud_samples),
            'V3': np.random.normal(1.5, 1.3, fraud_samples),
            'V4': np.random.normal(-0.5, 1.1, fraud_samples),
            'V5': np.random.normal(0.8, 1.4, fraud_samples),
            'Class': np.ones(fraud_samples)
        }
        
        # Combine datasets
        self.data = pd.DataFrame({
            'Amount': np.concatenate([normal_data['Amount'], fraud_data['Amount']]),
            'Time': np.concatenate([normal_data['Time'], fraud_data['Time']]),
            'V1': np.concatenate([normal_data['V1'], fraud_data['V1']]),
            'V2': np.concatenate([normal_data['V2'], fraud_data['V2']]),
            'V3': np.concatenate([normal_data['V3'], fraud_data['V3']]),
            'V4': np.concatenate([normal_data['V4'], fraud_data['V4']]),
            'V5': np.concatenate([normal_data['V5'], fraud_data['V5']]),
            'Class': np.concatenate([normal_data['Class'], fraud_data['Class']])
        })
        
        # Shuffle the data
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        # Feature engineering
        self.data['Hour'] = (self.data['Time'] / 3600) % 24
        self.data['Amount_log'] = np.log1p(self.data['Amount'])
        
        print(f"✅ Dataset loaded: {len(self.data)} transactions")
        print(f"   - Normal transactions: {len(self.data[self.data['Class'] == 0])}")
        print(f"   - Fraudulent transactions: {len(self.data[self.data['Class'] == 1])}")
        
    def exploratory_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("📊 Performing Exploratory Data Analysis...")
        
        # Basic statistics
        fraud_stats = {
            'Total Transactions': len(self.data),
            'Fraud Rate': f"{self.data['Class'].mean():.2%}",
            'Avg Transaction Amount': f"${self.data['Amount'].mean():.2f}",
            'Avg Fraud Amount': f"${self.data[self.data['Class'] == 1]['Amount'].mean():.2f}",
            'Total Fraud Loss': f"${self.data[self.data['Class'] == 1]['Amount'].sum():.2f}"
        }
        
        return fraud_stats
        
    def create_fraud_overview_dashboard(self):
        """Create comprehensive fraud overview dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Transaction Distribution', 'Amount Distribution by Class', 
                          'Fraud by Hour of Day', 'Feature Correlation', 
                          'Transaction Amount vs Time', 'Class Balance'),
            specs=[[{"type": "pie"}, {"type": "box"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Transaction distribution
        class_counts = self.data['Class'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Normal', 'Fraud'], values=class_counts.values, name="Transactions"),
            row=1, col=1
        )
        
        # Amount distribution by class
        for class_val in [0, 1]:
            class_data = self.data[self.data['Class'] == class_val]
            fig.add_trace(
                go.Box(y=class_data['Amount'], name=f"Class {class_val}"),
                row=1, col=2
            )
        
        # Fraud by hour
        hourly_fraud = self.data.groupby('Hour')['Class'].mean()
        fig.add_trace(
            go.Bar(x=hourly_fraud.index, y=hourly_fraud.values, name="Fraud Rate by Hour"),
            row=1, col=3
        )
        
        # Feature correlation
        corr_matrix = self.data[['V1', 'V2', 'V3', 'V4', 'V5', 'Amount_log', 'Class']].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns),
            row=2, col=1
        )
        
        # Amount vs Time scatter
        sample_data = self.data.sample(1000)  # Sample for performance
        fig.add_trace(
            go.Scatter(
                x=sample_data['Time'], 
                y=sample_data['Amount'],
                mode='markers',
                marker=dict(color=sample_data['Class'], colorscale='Viridis'),
                name="Transactions"
            ),
            row=2, col=2
        )
        
        # Class balance
        fig.add_trace(
            go.Bar(x=['Normal', 'Fraud'], y=class_counts.values, name="Count"),
            row=2, col=3
        )
        
        fig.update_layout(height=800, title_text="Credit Card Fraud Detection - Overview Dashboard")
        return fig
        
    def build_models(self):
        """Build and train multiple fraud detection models"""
        print("🤖 Building Fraud Detection Models...")
        
        # Prepare features
        features = ['Amount_log', 'V1', 'V2', 'V3', 'V4', 'V5', 'Hour']
        X = self.data[features]
        y = self.data['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle imbalanced data with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Model 1: Logistic Regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train_balanced, y_train_balanced)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Model 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_balanced, y_train_balanced)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Model 3: Isolation Forest (Anomaly Detection)
        iso_model = IsolationForest(contamination=0.01, random_state=42)
        iso_model.fit(X_train_scaled)
        iso_pred = iso_model.predict(X_test_scaled)
        iso_pred = np.where(iso_pred == -1, 1, 0)  # Convert to binary classification
        
        # Store models and results
        self.models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'Isolation Forest': iso_model,
            'scaler': scaler
        }
        
        self.results = {
            'Logistic Regression': {
                'predictions': lr_pred,
                'probabilities': lr_prob,
                'auc': roc_auc_score(y_test, lr_prob)
            },
            'Random Forest': {
                'predictions': rf_pred,
                'probabilities': rf_prob,
                'auc': roc_auc_score(y_test, rf_prob)
            },
            'Isolation Forest': {
                'predictions': iso_pred,
                'probabilities': None,
                'auc': roc_auc_score(y_test, iso_pred)
            }
        }
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("✅ Models trained successfully!")
        
    def create_model_performance_dashboard(self):
        """Create model performance comparison dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curves', 'Model AUC Comparison', 
                          'Confusion Matrix - Random Forest', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # ROC Curves
        for model_name in ['Logistic Regression', 'Random Forest']:
            if self.results[model_name]['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, self.results[model_name]['probabilities'])
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr, 
                        name=f"{model_name} (AUC: {self.results[model_name]['auc']:.3f})",
                        mode='lines'
                    ),
                    row=1, col=1
                )
        
        # Diagonal line for random classifier
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')),
            row=1, col=1
        )
        
        # AUC Comparison
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=auc_scores, name="AUC Score"),
            row=1, col=2
        )
        
        # Confusion Matrix for Random Forest
        cm = confusion_matrix(self.y_test, self.results['Random Forest']['predictions'])
        fig.add_trace(
            go.Heatmap(
                z=cm, 
                x=['Predicted Normal', 'Predicted Fraud'],
                y=['Actual Normal', 'Actual Fraud'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=2, col=1
        )
        
        # Feature Importance
        feature_names = ['Amount_log', 'V1', 'V2', 'V3', 'V4', 'V5', 'Hour']
        importance = self.models['Random Forest'].feature_importances_
        fig.add_trace(
            go.Bar(x=feature_names, y=importance, name="Feature Importance"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Fraud Detection Model Performance")
        return fig
        
    def create_business_impact_analysis(self):
        """Analyze business impact of fraud detection"""
        # Calculate business metrics
        fraud_transactions = self.data[self.data['Class'] == 1]
        total_fraud_amount = fraud_transactions['Amount'].sum()
        avg_fraud_amount = fraud_transactions['Amount'].mean()
        
        # Model performance on test set
        rf_predictions = self.results['Random Forest']['predictions']
        
        # Calculate detection rates
        true_positives = np.sum((self.y_test == 1) & (rf_predictions == 1))
        false_negatives = np.sum((self.y_test == 1) & (rf_predictions == 0))
        false_positives = np.sum((self.y_test == 0) & (rf_predictions == 1))
        
        detection_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / len(self.y_test[self.y_test == 0])
        
        # Estimate prevented losses
        prevented_loss = detection_rate * total_fraud_amount
        
        business_metrics = {
            'Total Fraud Amount': f"${total_fraud_amount:,.2f}",
            'Average Fraud Amount': f"${avg_fraud_amount:.2f}",
            'Detection Rate': f"{detection_rate:.1%}",
            'False Positive Rate': f"{false_positive_rate:.2%}",
            'Estimated Prevented Loss': f"${prevented_loss:,.2f}",
            'Model Accuracy': f"{roc_auc_score(self.y_test, self.results['Random Forest']['probabilities']):.3f}"
        }
        
        return business_metrics
        
    def generate_fraud_risk_scores(self):
        """Generate risk scores for new transactions"""
        # Use Random Forest model to generate risk scores
        risk_scores = self.models['Random Forest'].predict_proba(self.X_test)[:, 1]
        
        # Create risk categories
        risk_categories = pd.cut(risk_scores, 
                               bins=[0, 0.1, 0.3, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High', 'Critical'])
        
        risk_distribution = pd.DataFrame({
            'Risk_Score': risk_scores,
            'Risk_Category': risk_categories,
            'Actual_Class': self.y_test
        })
        
        return risk_distribution
        
    def run_complete_analysis(self):
        """Run complete fraud detection analysis"""
        print("🚀 Starting Complete Fraud Detection Analysis...")
        
        # Exploratory analysis
        fraud_stats = self.exploratory_analysis()
        
        # Build models
        self.build_models()
        
        # Business impact analysis
        business_impact = self.create_business_impact_analysis()
        
        # Generate visualizations
        overview_fig = self.create_fraud_overview_dashboard()
        performance_fig = self.create_model_performance_dashboard()
        
        # Risk scoring
        risk_scores = self.generate_fraud_risk_scores()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        overview_fig.write_html('results/fraud_overview_dashboard.html')
        performance_fig.write_html('results/model_performance_dashboard.html')
        
        # Save risk scores
        risk_scores.to_csv('results/fraud_risk_scores.csv', index=False)
        
        print("✅ Analysis Complete!")
        print("\n📊 Key Fraud Statistics:")
        for key, value in fraud_stats.items():
            print(f"   • {key}: {value}")
            
        print("\n💼 Business Impact Metrics:")
        for key, value in business_impact.items():
            print(f"   • {key}: {value}")
            
        print("\n📁 Files Generated:")
        print("   • results/fraud_overview_dashboard.html")
        print("   • results/model_performance_dashboard.html")
        print("   • results/fraud_risk_scores.csv")
        
        return fraud_stats, business_impact

if __name__ == "__main__":
    analyzer = FraudDetectionAnalyzer()
    fraud_stats, business_impact = analyzer.run_complete_analysis()