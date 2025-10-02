"""
KPI Root Cause Analysis & Performance Monitoring System
Advanced analytics for KPI monitoring and anomaly detection
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class KPIMonitoringSystem:
    def __init__(self):
        self.kpi_data = None
        self.anomalies = []
        
    def generate_kpi_data(self, n_days=365):
        """Generate realistic KPI dataset with anomalies"""
        np.random.seed(42)
        
        date_range = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Define KPIs to monitor
        kpis = [
            'Revenue', 'Website_Traffic', 'Conversion_Rate', 'Customer_Acquisition_Cost',
            'Customer_Lifetime_Value', 'Churn_Rate', 'Average_Order_Value', 'Page_Load_Time',
            'Customer_Satisfaction', 'Employee_Productivity', 'Inventory_Turnover', 'Profit_Margin'
        ]
        
        kpi_data = []
        
        for date in date_range:
            day_of_year = date.timetuple().tm_yday
            day_of_week = date.weekday()
            
            # Seasonal patterns
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly patterns (lower on weekends for some KPIs)
            weekly_factor = 0.7 if day_of_week >= 5 else 1.0
            
            for kpi in kpis:
                # Base values for each KPI
                base_values = {
                    'Revenue': 50000, 'Website_Traffic': 10000, 'Conversion_Rate': 0.05,
                    'Customer_Acquisition_Cost': 100, 'Customer_Lifetime_Value': 500,
                    'Churn_Rate': 0.02, 'Average_Order_Value': 150, 'Page_Load_Time': 2.5,
                    'Customer_Satisfaction': 4.2, 'Employee_Productivity': 85,
                    'Inventory_Turnover': 12, 'Profit_Margin': 0.15
                }
                
                base_value = base_values[kpi]
                
                # Apply patterns
                if kpi in ['Revenue', 'Website_Traffic', 'Average_Order_Value']:
                    value = base_value * seasonal_factor * weekly_factor
                else:
                    value = base_value * seasonal_factor
                
                # Add normal variation
                noise_factor = np.random.normal(1, 0.1)
                value *= noise_factor
                
                # Inject anomalies (5% chance)
                is_anomaly = False
                if np.random.random() < 0.05:
                    anomaly_type = np.random.choice(['spike', 'drop'])
                    if anomaly_type == 'spike':
                        value *= np.random.uniform(1.5, 3.0)
                    else:
                        value *= np.random.uniform(0.3, 0.7)
                    is_anomaly = True
                
                # Ensure positive values and reasonable ranges
                if kpi in ['Conversion_Rate', 'Churn_Rate', 'Profit_Margin']:
                    value = max(0, min(1, value))
                elif kpi == 'Customer_Satisfaction':
                    value = max(1, min(5, value))
                else:
                    value = max(0, value)
                
                kpi_data.append({
                    'date': date,
                    'kpi_name': kpi,
                    'value': value,
                    'baseline_value': base_value,
                    'is_anomaly': is_anomaly,
                    'day_of_week': day_of_week,
                    'seasonal_factor': seasonal_factor
                })
        
        self.data = pd.DataFrame(kpi_data)
        
        print(f"✅ Generated {len(self.data):,} KPI data points")
        print(f"   - KPIs monitored: {len(kpis)}")
        print(f"   - Date range: {date_range[0]} to {date_range[-1]}")
        print(f"   - Anomalies injected: {self.data['is_anomaly'].sum()}")
        
        return self.data
    
    def detect_anomalies(self):
        """Detect anomalies using statistical methods"""
        print("🔄 Detecting anomalies...")
        
        detected_anomalies = []
        
        for kpi in self.data['kpi_name'].unique():
            kpi_data = self.data[self.data['kpi_name'] == kpi].copy()
            kpi_data = kpi_data.sort_values('date')
            
            # Calculate rolling statistics
            window = 7  # 7-day window
            kpi_data['rolling_mean'] = kpi_data['value'].rolling(window=window).mean()
            kpi_data['rolling_std'] = kpi_data['value'].rolling(window=window).std()
            
            # Z-score based anomaly detection
            kpi_data['z_score'] = abs((kpi_data['value'] - kpi_data['rolling_mean']) / kpi_data['rolling_std'])
            
            # Identify anomalies (z-score > 2.5)
            anomaly_threshold = 2.5
            anomalies = kpi_data[kpi_data['z_score'] > anomaly_threshold]
            
            for _, anomaly in anomalies.iterrows():
                detected_anomalies.append({
                    'date': anomaly['date'],
                    'kpi_name': kpi,
                    'value': anomaly['value'],
                    'expected_value': anomaly['rolling_mean'],
                    'z_score': anomaly['z_score'],
                    'severity': 'High' if anomaly['z_score'] > 3 else 'Medium',
                    'actual_anomaly': anomaly['is_anomaly']  # For validation
                })
        
        self.detected_anomalies = pd.DataFrame(detected_anomalies)
        
        # Calculate detection accuracy
        if len(self.detected_anomalies) > 0:
            true_positives = self.detected_anomalies['actual_anomaly'].sum()
            detection_rate = true_positives / len(self.detected_anomalies)
        else:
            detection_rate = 0
        
        print(f"✅ Anomaly detection complete!")
        print(f"   - Anomalies detected: {len(self.detected_anomalies)}")
        print(f"   - Detection accuracy: {detection_rate:.1%}")
    
    def perform_root_cause_analysis(self):
        """Perform root cause analysis for detected anomalies"""
        print("🔄 Performing root cause analysis...")
        
        root_cause_analysis = []
        
        for _, anomaly in self.detected_anomalies.iterrows():
            date = anomaly['date']
            kpi_name = anomaly['kpi_name']
            
            # Analyze potential causes
            causes = []
            
            # Check if it's a weekend effect
            if pd.to_datetime(date).weekday() >= 5:
                causes.append("Weekend Effect")
            
            # Check for seasonal patterns
            day_of_year = pd.to_datetime(date).timetuple().tm_yday
            if day_of_year in range(355, 366) or day_of_year in range(1, 15):  # Holiday season
                causes.append("Holiday Season")
            
            # Check for correlated anomalies on the same date
            same_date_anomalies = self.detected_anomalies[self.detected_anomalies['date'] == date]
            if len(same_date_anomalies) > 1:
                causes.append("System-wide Issue")
            
            # KPI-specific causes
            if kpi_name in ['Revenue', 'Website_Traffic']:
                causes.append("Marketing Campaign Impact")
            elif kpi_name in ['Page_Load_Time', 'Customer_Satisfaction']:
                causes.append("Technical Issue")
            elif kpi_name in ['Churn_Rate', 'Customer_Acquisition_Cost']:
                causes.append("Competitive Pressure")
            
            if not causes:
                causes.append("Unknown - Requires Investigation")
            
            root_cause_analysis.append({
                'date': date,
                'kpi_name': kpi_name,
                'anomaly_value': anomaly['value'],
                'severity': anomaly['severity'],
                'potential_causes': ', '.join(causes),
                'recommended_actions': self._get_recommended_actions(kpi_name, causes)
            })
        
        self.root_cause_analysis = pd.DataFrame(root_cause_analysis)
        
        print("✅ Root cause analysis complete!")
    
    def _get_recommended_actions(self, kpi_name, causes):
        """Get recommended actions based on KPI and causes"""
        actions = []
        
        if "Weekend Effect" in causes:
            actions.append("Review weekend operations")
        
        if "Holiday Season" in causes:
            actions.append("Adjust seasonal forecasts")
        
        if "System-wide Issue" in causes:
            actions.append("Check system infrastructure")
        
        if kpi_name == 'Revenue' and "Marketing Campaign Impact" in causes:
            actions.append("Analyze campaign performance")
        
        if kpi_name == 'Page_Load_Time' and "Technical Issue" in causes:
            actions.append("Investigate server performance")
        
        if not actions:
            actions.append("Conduct detailed investigation")
        
        return ', '.join(actions)
    
    def run_complete_analysis(self):
        """Run complete KPI monitoring analysis"""
        print("🚀 Starting KPI Monitoring & Root Cause Analysis...")
        
        # Generate and analyze data
        self.generate_kpi_data()
        self.detect_anomalies()
        self.perform_root_cause_analysis()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/kpi_data.csv', index=False)
        self.detected_anomalies.to_csv('results/detected_anomalies.csv', index=False)
        self.root_cause_analysis.to_csv('results/root_cause_analysis.csv', index=False)
        
        # Generate summary statistics
        summary_stats = {
            'total_kpi_datapoints': len(self.data),
            'kpis_monitored': self.data['kpi_name'].nunique(),
            'anomalies_detected': len(self.detected_anomalies),
            'high_severity_anomalies': len(self.detected_anomalies[self.detected_anomalies['severity'] == 'High']),
            'most_anomalous_kpi': self.detected_anomalies['kpi_name'].value_counts().index[0] if len(self.detected_anomalies) > 0 else 'None'
        }
        
        import json
        with open('results/kpi_monitoring_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print("\n✅ KPI Monitoring Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/kpi_data.csv")
        print("   - results/detected_anomalies.csv")
        print("   - results/root_cause_analysis.csv")
        print("   - results/kpi_monitoring_summary.json")
        
        return {
            'status': 'complete',
            'kpis_monitored': self.data['kpi_name'].nunique(),
            'anomalies_detected': len(self.detected_anomalies),
            'high_severity_anomalies': len(self.detected_anomalies[self.detected_anomalies['severity'] == 'High'])
        }

if __name__ == "__main__":
    kpi_system = KPIMonitoringSystem()
    results = kpi_system.run_complete_analysis()