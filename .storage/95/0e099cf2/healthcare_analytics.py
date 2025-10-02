"""
Hospital Patient Readmission Prediction System
Healthcare analytics for predicting patient readmission risk
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class HealthcareAnalyticsSystem:
    def __init__(self):
        self.risk_model = None
        
    def generate_patient_data(self, n_patients=25000):
        """Generate realistic patient dataset"""
        np.random.seed(42)
        
        patients = []
        for i in range(n_patients):
            # Patient demographics
            age = np.random.normal(65, 15)
            age = max(18, min(95, age))  # Constrain age
            
            # Medical conditions
            diabetes = np.random.choice([0, 1], p=[0.7, 0.3])
            hypertension = np.random.choice([0, 1], p=[0.6, 0.4])
            heart_disease = np.random.choice([0, 1], p=[0.8, 0.2])
            
            # Hospital stay details
            length_of_stay = np.random.poisson(5) + 1
            num_procedures = np.random.poisson(2)
            num_medications = np.random.poisson(8) + 1
            
            # Calculate readmission risk
            risk_score = (
                (age - 65) * 0.01 +
                diabetes * 0.15 +
                hypertension * 0.10 +
                heart_disease * 0.20 +
                length_of_stay * 0.02 +
                num_procedures * 0.05 +
                (num_medications - 8) * 0.01
            )
            
            # Add noise and convert to probability
            risk_score += np.random.normal(0, 0.1)
            readmission_prob = 1 / (1 + np.exp(-risk_score))  # Sigmoid
            
            # Determine actual readmission
            readmitted = np.random.random() < readmission_prob
            
            patients.append({
                'patient_id': f'P{i+1:06d}',
                'age': int(age),
                'gender': np.random.choice(['M', 'F']),
                'diabetes': diabetes,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'length_of_stay': length_of_stay,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'discharge_date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_patients)[i],
                'readmission_risk_score': risk_score,
                'readmitted_30_days': int(readmitted)
            })
        
        self.data = pd.DataFrame(patients)
        
        print(f"✅ Generated {len(self.data):,} patient records")
        print(f"   - Readmission rate: {self.data['readmitted_30_days'].mean():.1%}")
        print(f"   - Average age: {self.data['age'].mean():.1f}")
        print(f"   - Average length of stay: {self.data['length_of_stay'].mean():.1f} days")
        
        return self.data
    
    def analyze_readmission_factors(self):
        """Analyze factors contributing to readmission"""
        print("🔄 Analyzing readmission factors...")
        
        # Calculate readmission rates by factor
        self.factor_analysis = {}
        
        # Age groups
        self.data['age_group'] = pd.cut(self.data['age'], bins=[0, 50, 65, 80, 100], 
                                       labels=['<50', '50-65', '65-80', '80+'])
        age_readmission = self.data.groupby('age_group')['readmitted_30_days'].mean()
        self.factor_analysis['age_groups'] = age_readmission.to_dict()
        
        # Medical conditions
        conditions = ['diabetes', 'hypertension', 'heart_disease']
        for condition in conditions:
            condition_readmission = self.data.groupby(condition)['readmitted_30_days'].mean()
            self.factor_analysis[condition] = condition_readmission.to_dict()
        
        # Length of stay
        self.data['los_group'] = pd.cut(self.data['length_of_stay'], bins=[0, 3, 7, 14, 100], 
                                       labels=['1-3', '4-7', '8-14', '15+'])
        los_readmission = self.data.groupby('los_group')['readmitted_30_days'].mean()
        self.factor_analysis['length_of_stay'] = los_readmission.to_dict()
        
        print("✅ Factor analysis complete!")
        
    def create_risk_stratification(self):
        """Create patient risk stratification"""
        print("🔄 Creating risk stratification...")
        
        # Define risk categories based on risk score
        risk_thresholds = self.data['readmission_risk_score'].quantile([0.33, 0.67])
        
        def categorize_risk(score):
            if score < risk_thresholds.iloc[0]:
                return 'Low Risk'
            elif score < risk_thresholds.iloc[1]:
                return 'Medium Risk'
            else:
                return 'High Risk'
        
        self.data['risk_category'] = self.data['readmission_risk_score'].apply(categorize_risk)
        
        # Calculate actual readmission rates by risk category
        risk_performance = self.data.groupby('risk_category')['readmitted_30_days'].agg(['count', 'mean'])
        
        self.risk_stratification = {
            'Low Risk': {
                'patients': int(risk_performance.loc['Low Risk', 'count']),
                'readmission_rate': risk_performance.loc['Low Risk', 'mean']
            },
            'Medium Risk': {
                'patients': int(risk_performance.loc['Medium Risk', 'count']),
                'readmission_rate': risk_performance.loc['Medium Risk', 'mean']
            },
            'High Risk': {
                'patients': int(risk_performance.loc['High Risk', 'count']),
                'readmission_rate': risk_performance.loc['High Risk', 'mean']
            }
        }
        
        print("✅ Risk stratification complete!")
    
    def run_complete_analysis(self):
        """Run complete healthcare analytics"""
        print("🚀 Starting Hospital Readmission Analysis...")
        
        # Generate and analyze data
        self.generate_patient_data()
        self.analyze_readmission_factors()
        self.create_risk_stratification()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/patient_data.csv', index=False)
        
        import json
        with open('results/healthcare_analysis.json', 'w') as f:
            json.dump({
                'factor_analysis': self.factor_analysis,
                'risk_stratification': self.risk_stratification
            }, f, indent=2, default=str)
        
        print("\n✅ Healthcare Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/patient_data.csv")
        print("   - results/healthcare_analysis.json")
        
        return {
            'status': 'complete',
            'patients': len(self.data),
            'readmission_rate': f"{self.data['readmitted_30_days'].mean():.1%}"
        }

if __name__ == "__main__":
    healthcare_system = HealthcareAnalyticsSystem()
    results = healthcare_system.run_complete_analysis()