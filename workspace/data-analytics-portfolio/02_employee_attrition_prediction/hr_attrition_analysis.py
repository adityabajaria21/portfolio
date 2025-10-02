"""
Employee Attrition Prediction Analysis
HR Analytics with predictive modeling and business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class HRAttritionAnalyzer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Generate comprehensive HR dataset"""
        np.random.seed(42)
        n_employees = 2000
        
        # Generate employee data
        departments = ['Sales', 'R&D', 'HR', 'Marketing', 'IT', 'Finance']
        job_roles = ['Manager', 'Senior', 'Mid-level', 'Junior', 'Intern']
        education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        
        self.data = pd.DataFrame({
            'EmployeeID': range(1, n_employees + 1),
            'Age': np.random.normal(35, 8, n_employees).astype(int),
            'Department': np.random.choice(departments, n_employees),
            'JobRole': np.random.choice(job_roles, n_employees),
            'Education': np.random.choice(education_levels, n_employees),
            'YearsAtCompany': np.random.exponential(3, n_employees).astype(int),
            'YearsInCurrentRole': np.random.exponential(2, n_employees).astype(int),
            'MonthlyIncome': np.random.normal(5000, 2000, n_employees),
            'JobSatisfaction': np.random.randint(1, 5, n_employees),
            'WorkLifeBalance': np.random.randint(1, 5, n_employees),
            'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
            'DistanceFromHome': np.random.exponential(10, n_employees),
            'NumCompaniesWorked': np.random.poisson(2, n_employees),
            'TrainingTimesLastYear': np.random.poisson(3, n_employees),
            'PerformanceRating': np.random.randint(1, 5, n_employees)
        })
        
        # Ensure realistic constraints
        self.data['Age'] = np.clip(self.data['Age'], 18, 65)
        self.data['YearsAtCompany'] = np.clip(self.data['YearsAtCompany'], 0, 40)
        self.data['YearsInCurrentRole'] = np.minimum(self.data['YearsInCurrentRole'], self.data['YearsAtCompany'])
        self.data['MonthlyIncome'] = np.clip(self.data['MonthlyIncome'], 2000, 15000)
        
        # Create attrition based on logical factors
        attrition_probability = (
            0.1 +  # Base rate
            0.2 * (self.data['JobSatisfaction'] <= 2) +  # Low job satisfaction
            0.15 * (self.data['WorkLifeBalance'] <= 2) +  # Poor work-life balance
            0.1 * (self.data['OverTime'] == 'Yes') +  # Overtime
            0.1 * (self.data['MonthlyIncome'] < 3000) +  # Low income
            0.05 * (self.data['DistanceFromHome'] > 20) +  # Long commute
            0.1 * (self.data['YearsAtCompany'] < 1) +  # New employees
            0.05 * (self.data['NumCompaniesWorked'] > 4)  # Job hoppers
        )
        
        # Cap probability at 0.8
        attrition_probability = np.minimum(attrition_probability, 0.8)
        
        # Generate attrition based on probability
        self.data['Attrition'] = np.random.binomial(1, attrition_probability, n_employees)
        
        # Feature engineering
        self.data['IncomePerYear'] = self.data['MonthlyIncome'] * 12
        self.data['ExperienceRatio'] = self.data['YearsInCurrentRole'] / (self.data['YearsAtCompany'] + 1)
        self.data['SatisfactionScore'] = (self.data['JobSatisfaction'] + self.data['WorkLifeBalance']) / 2
        
        print(f"✅ HR Dataset generated: {len(self.data)} employees")
        print(f"   - Attrition rate: {self.data['Attrition'].mean():.1%}")
        
    def exploratory_analysis(self):
        """Comprehensive HR exploratory analysis"""
        print("📊 Performing HR Exploratory Analysis...")
        
        # Key HR metrics
        hr_metrics = {
            'Total Employees': len(self.data),
            'Attrition Rate': f"{self.data['Attrition'].mean():.1%}",
            'Average Age': f"{self.data['Age'].mean():.1f} years",
            'Average Tenure': f"{self.data['YearsAtCompany'].mean():.1f} years",
            'Average Income': f"${self.data['MonthlyIncome'].mean():,.0f}",
            'Overtime Employees': f"{(self.data['OverTime'] == 'Yes').mean():.1%}"
        }
        
        # Attrition by department
        dept_attrition = self.data.groupby('Department')['Attrition'].agg(['count', 'sum', 'mean']).round(3)
        dept_attrition.columns = ['Total_Employees', 'Attritions', 'Attrition_Rate']
        
        return hr_metrics, dept_attrition
        
    def create_hr_overview_dashboard(self):
        """Create comprehensive HR overview dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Attrition by Department', 'Age Distribution by Attrition', 
                          'Income vs Satisfaction', 'Attrition by Job Role',
                          'Years at Company Distribution', 'Work-Life Balance Impact'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Attrition by department
        dept_stats = self.data.groupby('Department')['Attrition'].mean()
        fig.add_trace(
            go.Bar(x=dept_stats.index, y=dept_stats.values, name="Attrition Rate"),
            row=1, col=1
        )
        
        # Age distribution by attrition
        for attrition in [0, 1]:
            age_data = self.data[self.data['Attrition'] == attrition]['Age']
            fig.add_trace(
                go.Histogram(x=age_data, name=f"Attrition: {attrition}", opacity=0.7),
                row=1, col=2
            )
        
        # Income vs Satisfaction scatter
        fig.add_trace(
            go.Scatter(
                x=self.data['MonthlyIncome'],
                y=self.data['SatisfactionScore'],
                mode='markers',
                marker=dict(color=self.data['Attrition'], colorscale='Viridis'),
                name="Employees"
            ),
            row=2, col=1
        )
        
        # Attrition by job role
        role_stats = self.data.groupby('JobRole')['Attrition'].mean()
        fig.add_trace(
            go.Bar(x=role_stats.index, y=role_stats.values, name="Attrition Rate"),
            row=2, col=2
        )
        
        # Years at company distribution
        fig.add_trace(
            go.Histogram(x=self.data['YearsAtCompany'], name="Years at Company"),
            row=3, col=1
        )
        
        # Work-life balance impact
        wlb_stats = self.data.groupby('WorkLifeBalance')['Attrition'].mean()
        fig.add_trace(
            go.Bar(x=wlb_stats.index, y=wlb_stats.values, name="Attrition Rate"),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, title_text="HR Analytics - Employee Attrition Overview")
        return fig
        
    def build_attrition_models(self):
        """Build predictive models for employee attrition"""
        print("🤖 Building Attrition Prediction Models...")
        
        # Prepare features
        # Encode categorical variables
        le_dept = LabelEncoder()
        le_role = LabelEncoder()
        le_edu = LabelEncoder()
        le_ot = LabelEncoder()
        
        features_df = self.data.copy()
        features_df['Department_encoded'] = le_dept.fit_transform(features_df['Department'])
        features_df['JobRole_encoded'] = le_role.fit_transform(features_df['JobRole'])
        features_df['Education_encoded'] = le_edu.fit_transform(features_df['Education'])
        features_df['OverTime_encoded'] = le_ot.fit_transform(features_df['OverTime'])
        
        # Select features
        feature_columns = [
            'Age', 'Department_encoded', 'JobRole_encoded', 'Education_encoded',
            'YearsAtCompany', 'YearsInCurrentRole', 'MonthlyIncome',
            'JobSatisfaction', 'WorkLifeBalance', 'OverTime_encoded',
            'DistanceFromHome', 'NumCompaniesWorked', 'TrainingTimesLastYear',
            'PerformanceRating', 'SatisfactionScore', 'ExperienceRatio'
        ]
        
        X = features_df[feature_columns]
        y = features_df['Attrition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model 1: Logistic Regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Model 2: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        # Model 3: Gradient Boosting
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_prob = gb_model.predict_proba(X_test)[:, 1]
        
        # Store models and results
        self.models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model,
            'scaler': scaler,
            'encoders': {
                'department': le_dept,
                'job_role': le_role,
                'education': le_edu,
                'overtime': le_ot
            }
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
            'Gradient Boosting': {
                'predictions': gb_pred,
                'probabilities': gb_prob,
                'auc': roc_auc_score(y_test, gb_prob)
            }
        }
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_columns
        
        print("✅ Attrition models trained successfully!")
        
    def create_model_performance_dashboard(self):
        """Create model performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model AUC Comparison', 'Feature Importance (Random Forest)', 
                          'Attrition Risk Distribution', 'Top Risk Factors'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Model AUC comparison
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=auc_scores, name="AUC Score"),
            row=1, col=1
        )
        
        # Feature importance
        importance = self.models['Random Forest'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(x=feature_importance['Importance'], y=feature_importance['Feature'], 
                   orientation='h', name="Feature Importance"),
            row=1, col=2
        )
        
        # Risk distribution
        rf_prob = self.results['Random Forest']['probabilities']
        fig.add_trace(
            go.Histogram(x=rf_prob, nbinsx=20, name="Attrition Risk Score"),
            row=2, col=1
        )
        
        # Top risk factors analysis
        high_risk_employees = self.X_test[rf_prob > 0.7]
        if len(high_risk_employees) > 0:
            avg_features = high_risk_employees.mean()
            top_features = avg_features.nlargest(8)
            fig.add_trace(
                go.Bar(x=top_features.index, y=top_features.values, name="High Risk Profile"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Employee Attrition - Model Performance")
        return fig
        
    def create_retention_strategy_dashboard(self):
        """Create retention strategy recommendations"""
        # Analyze high-risk employees
        rf_prob = self.results['Random Forest']['probabilities']
        high_risk_mask = rf_prob > 0.6
        
        if np.sum(high_risk_mask) > 0:
            high_risk_data = self.data.iloc[self.X_test.index[high_risk_mask]]
            
            # Retention strategies analysis
            retention_analysis = {
                'High Risk Employees': len(high_risk_data),
                'Avg Risk Score': f"{rf_prob[high_risk_mask].mean():.2f}",
                'Most At-Risk Department': high_risk_data['Department'].mode().iloc[0],
                'Avg Satisfaction Score': f"{high_risk_data['SatisfactionScore'].mean():.2f}",
                'Overtime Percentage': f"{(high_risk_data['OverTime'] == 'Yes').mean():.1%}",
                'Avg Years at Company': f"{high_risk_data['YearsAtCompany'].mean():.1f}"
            }
            
            # Cost analysis
            avg_salary = high_risk_data['MonthlyIncome'].mean() * 12
            replacement_cost = avg_salary * 0.5  # Assume 50% of annual salary as replacement cost
            total_potential_loss = len(high_risk_data) * replacement_cost
            
            cost_analysis = {
                'Potential Annual Loss': f"${total_potential_loss:,.0f}",
                'Avg Replacement Cost': f"${replacement_cost:,.0f}",
                'High Risk Employee Salaries': f"${high_risk_data['MonthlyIncome'].sum() * 12:,.0f}"
            }
            
            return retention_analysis, cost_analysis
        
        return {}, {}
        
    def run_complete_analysis(self):
        """Run complete HR attrition analysis"""
        print("🚀 Starting Complete HR Attrition Analysis...")
        
        # Exploratory analysis
        hr_metrics, dept_attrition = self.exploratory_analysis()
        
        # Build models
        self.build_attrition_models()
        
        # Retention strategy analysis
        retention_analysis, cost_analysis = self.create_retention_strategy_dashboard()
        
        # Generate visualizations
        overview_fig = self.create_hr_overview_dashboard()
        performance_fig = self.create_model_performance_dashboard()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        overview_fig.write_html('results/hr_overview_dashboard.html')
        performance_fig.write_html('results/hr_model_performance.html')
        
        # Save department analysis
        dept_attrition.to_csv('results/department_attrition_analysis.csv')
        
        # Generate employee risk scores
        rf_prob = self.results['Random Forest']['probabilities']
        risk_scores = pd.DataFrame({
            'EmployeeID': self.data.iloc[self.X_test.index]['EmployeeID'],
            'Attrition_Risk_Score': rf_prob,
            'Risk_Category': pd.cut(rf_prob, bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                  labels=['Low', 'Medium', 'High', 'Critical'])
        })
        risk_scores.to_csv('results/employee_risk_scores.csv', index=False)
        
        print("✅ HR Analysis Complete!")
        print("\n📊 Key HR Metrics:")
        for key, value in hr_metrics.items():
            print(f"   • {key}: {value}")
            
        if retention_analysis:
            print("\n🎯 Retention Strategy Insights:")
            for key, value in retention_analysis.items():
                print(f"   • {key}: {value}")
                
        if cost_analysis:
            print("\n💰 Cost Impact Analysis:")
            for key, value in cost_analysis.items():
                print(f"   • {key}: {value}")
        
        print("\n📁 Files Generated:")
        print("   • results/hr_overview_dashboard.html")
        print("   • results/hr_model_performance.html")
        print("   • results/department_attrition_analysis.csv")
        print("   • results/employee_risk_scores.csv")
        
        return hr_metrics, retention_analysis, cost_analysis

if __name__ == "__main__":
    analyzer = HRAttritionAnalyzer()
    hr_metrics, retention_analysis, cost_analysis = analyzer.run_complete_analysis()