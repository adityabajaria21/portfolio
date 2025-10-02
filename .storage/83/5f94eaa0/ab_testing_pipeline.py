"""
A/B Testing Analysis System
Statistical testing framework for conversion optimization
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ABTestingSystem:
    def __init__(self):
        self.test_results = {}
        
    def generate_ab_test_data(self, n_users=10000):
        """Generate realistic A/B test dataset"""
        np.random.seed(42)
        
        # Test scenarios
        tests = [
            {'name': 'Homepage_Redesign', 'control_rate': 0.12, 'treatment_rate': 0.15},
            {'name': 'Checkout_Flow', 'control_rate': 0.08, 'treatment_rate': 0.11},
            {'name': 'Email_Campaign', 'control_rate': 0.05, 'treatment_rate': 0.07},
            {'name': 'Product_Page', 'control_rate': 0.18, 'treatment_rate': 0.22},
            {'name': 'Mobile_App', 'control_rate': 0.14, 'treatment_rate': 0.16}
        ]
        
        all_data = []
        
        for test in tests:
            test_users = n_users // len(tests)
            
            # Generate control group
            control_users = test_users // 2
            control_conversions = np.random.binomial(1, test['control_rate'], control_users)
            
            for i in range(control_users):
                all_data.append({
                    'user_id': f"U{len(all_data)+1:06d}",
                    'test_name': test['name'],
                    'variant': 'Control',
                    'converted': control_conversions[i],
                    'session_duration': np.random.lognormal(4, 0.5),  # minutes
                    'page_views': np.random.poisson(5),
                    'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.5, 0.4, 0.1]),
                    'traffic_source': np.random.choice(['Organic', 'Paid', 'Direct', 'Social'], p=[0.4, 0.3, 0.2, 0.1]),
                    'test_date': pd.date_range(start='2024-01-01', end='2024-02-01', periods=control_users)[i]
                })
            
            # Generate treatment group
            treatment_users = test_users - control_users
            treatment_conversions = np.random.binomial(1, test['treatment_rate'], treatment_users)
            
            for i in range(treatment_users):
                all_data.append({
                    'user_id': f"U{len(all_data)+1:06d}",
                    'test_name': test['name'],
                    'variant': 'Treatment',
                    'converted': treatment_conversions[i],
                    'session_duration': np.random.lognormal(4.1, 0.5),  # Slightly higher for treatment
                    'page_views': np.random.poisson(5.2),
                    'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.5, 0.4, 0.1]),
                    'traffic_source': np.random.choice(['Organic', 'Paid', 'Direct', 'Social'], p=[0.4, 0.3, 0.2, 0.1]),
                    'test_date': pd.date_range(start='2024-01-01', end='2024-02-01', periods=treatment_users)[i]
                })
        
        self.data = pd.DataFrame(all_data)
        
        print(f"✅ Generated {len(self.data):,} A/B test records")
        print(f"   - Tests: {len(tests)}")
        print(f"   - Date range: {self.data['test_date'].min()} to {self.data['test_date'].max()}")
        print(f"   - Overall conversion rate: {self.data['converted'].mean():.2%}")
        
        return self.data
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests for each A/B test"""
        print("🔄 Performing statistical tests...")
        
        self.test_results = {}
        
        for test_name in self.data['test_name'].unique():
            test_data = self.data[self.data['test_name'] == test_name]
            
            # Separate control and treatment groups
            control = test_data[test_data['variant'] == 'Control']
            treatment = test_data[test_data['variant'] == 'Treatment']
            
            # Calculate basic metrics
            control_conversions = control['converted'].sum()
            control_users = len(control)
            control_rate = control_conversions / control_users
            
            treatment_conversions = treatment['converted'].sum()
            treatment_users = len(treatment)
            treatment_rate = treatment_conversions / treatment_users
            
            # Calculate relative improvement
            relative_improvement = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
            
            # Perform Chi-square test
            contingency_table = np.array([
                [control_conversions, control_users - control_conversions],
                [treatment_conversions, treatment_users - treatment_conversions]
            ])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate confidence interval for difference in proportions
            se = np.sqrt((control_rate * (1 - control_rate) / control_users) + 
                        (treatment_rate * (1 - treatment_rate) / treatment_users))
            
            margin_of_error = 1.96 * se  # 95% confidence interval
            diff = treatment_rate - control_rate
            ci_lower = diff - margin_of_error
            ci_upper = diff + margin_of_error
            
            # Statistical power calculation (simplified)
            effect_size = abs(diff) / np.sqrt((control_rate * (1 - control_rate) + treatment_rate * (1 - treatment_rate)) / 2)
            
            # Determine significance
            is_significant = p_value < 0.05
            
            self.test_results[test_name] = {
                'control_users': control_users,
                'treatment_users': treatment_users,
                'control_conversions': control_conversions,
                'treatment_conversions': treatment_conversions,
                'control_rate': control_rate,
                'treatment_rate': treatment_rate,
                'relative_improvement': relative_improvement,
                'absolute_difference': diff,
                'p_value': p_value,
                'chi2_statistic': chi2,
                'is_significant': is_significant,
                'confidence_interval': (ci_lower, ci_upper),
                'effect_size': effect_size,
                'statistical_power': min(0.99, effect_size * 0.5 + 0.5)  # Simplified power calculation
            }
        
        significant_tests = sum(1 for result in self.test_results.values() if result['is_significant'])
        print(f"✅ Statistical analysis complete!")
        print(f"   - Tests analyzed: {len(self.test_results)}")
        print(f"   - Statistically significant: {significant_tests}")
        print(f"   - Average p-value: {np.mean([r['p_value'] for r in self.test_results.values()]):.4f}")
    
    def calculate_sample_size_recommendations(self, baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80):
        """Calculate recommended sample sizes for future tests"""
        # Simplified sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)  # 1.96 for 95% confidence
        z_beta = stats.norm.ppf(power)  # 0.84 for 80% power
        
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_pooled = (p1 + p2) / 2
        
        n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
        
        return int(np.ceil(n))
    
    def create_ab_testing_dashboard(self):
        """Create comprehensive A/B testing dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Conversion Rates by Test', 'Statistical Significance', 'Relative Improvement',
                          'P-values Distribution', 'Confidence Intervals', 'Sample Sizes',
                          'Conversion by Device', 'Test Timeline', 'Power Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        test_names = list(self.test_results.keys())
        
        # 1. Conversion Rates by Test
        control_rates = [self.test_results[test]['control_rate'] for test in test_names]
        treatment_rates = [self.test_results[test]['treatment_rate'] for test in test_names]
        
        fig.add_trace(
            go.Bar(x=test_names, y=control_rates, name='Control', marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=test_names, y=treatment_rates, name='Treatment', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. Statistical Significance
        significance = ['Significant' if self.test_results[test]['is_significant'] else 'Not Significant' for test in test_names]
        significance_counts = pd.Series(significance).value_counts()
        
        fig.add_trace(
            go.Bar(x=significance_counts.index, y=significance_counts.values, name='Significance'),
            row=1, col=2
        )
        
        # 3. Relative Improvement
        improvements = [self.test_results[test]['relative_improvement'] * 100 for test in test_names]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        fig.add_trace(
            go.Bar(x=test_names, y=improvements, name='Improvement %', marker_color=colors),
            row=1, col=3
        )
        
        # 4. P-values Distribution
        p_values = [self.test_results[test]['p_value'] for test in test_names]
        fig.add_trace(
            go.Histogram(x=p_values, nbinsx=10, name='P-values'),
            row=2, col=1
        )
        
        # Add significance line
        fig.add_vline(x=0.05, line_dash="dash", line_color="red", row=2, col=1)
        
        # 5. Confidence Intervals
        for i, test in enumerate(test_names):
            ci = self.test_results[test]['confidence_interval']
            fig.add_trace(
                go.Scatter(x=[ci[0], ci[1]], y=[i, i], mode='lines+markers', 
                          name=f'{test} CI', line=dict(width=3)),
                row=2, col=2
            )
        
        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        # 6. Sample Sizes
        sample_sizes = [self.test_results[test]['control_users'] + self.test_results[test]['treatment_users'] for test in test_names]
        fig.add_trace(
            go.Bar(x=test_names, y=sample_sizes, name='Sample Size'),
            row=2, col=3
        )
        
        # 7. Conversion by Device
        device_conversion = self.data.groupby('device_type')['converted'].mean()
        fig.add_trace(
            go.Bar(x=device_conversion.index, y=device_conversion.values, name='Device Conversion'),
            row=3, col=1
        )
        
        # 8. Test Timeline
        daily_conversions = self.data.groupby(self.data['test_date'].dt.date)['converted'].mean()
        fig.add_trace(
            go.Scatter(x=daily_conversions.index, y=daily_conversions.values, 
                      mode='lines+markers', name='Daily Conversion Rate'),
            row=3, col=2
        )
        
        # 9. Power Analysis
        statistical_power = [self.test_results[test]['statistical_power'] for test in test_names]
        fig.add_trace(
            go.Bar(x=test_names, y=statistical_power, name='Statistical Power'),
            row=3, col=3
        )
        
        # Add power threshold line
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=3)
        
        fig.update_layout(height=1200, title_text="A/B Testing Analysis Dashboard")
        return fig
    
    def generate_testing_recommendations(self):
        """Generate actionable A/B testing recommendations"""
        # Analyze results
        significant_tests = [test for test, result in self.test_results.items() if result['is_significant']]
        best_performing_test = max(self.test_results.items(), key=lambda x: x[1]['relative_improvement'])
        
        # Sample size recommendations
        recommended_sample_size = self.calculate_sample_size_recommendations()
        
        recommendations = {
            'test_results_summary': {
                'total_tests_analyzed': len(self.test_results),
                'statistically_significant_tests': len(significant_tests),
                'best_performing_test': best_performing_test[0],
                'best_improvement': f"{best_performing_test[1]['relative_improvement']:.2%}"
            },
            'immediate_actions': [
                f"Implement changes from {best_performing_test[0]} test (showed {best_performing_test[1]['relative_improvement']:.2%} improvement)",
                f"Roll out significant improvements from {len(significant_tests)} successful tests",
                "Stop tests that show no significant improvement after adequate sample size",
                "Investigate why some tests underperformed vs. expectations"
            ],
            'future_testing_strategy': [
                f"Use minimum sample size of {recommended_sample_size:,} users per variant for future tests",
                "Run tests for at least 2 weeks to account for weekly patterns",
                "Focus on high-impact areas like checkout flow and product pages",
                "Implement sequential testing to stop early winners/losers"
            ],
            'statistical_best_practices': [
                "Always pre-define success metrics before starting tests",
                "Use proper randomization to avoid selection bias",
                "Account for multiple testing when running concurrent experiments",
                "Monitor for novelty effects in the first few days of testing"
            ],
            'optimization_opportunities': [
                "Mobile conversion rates are lower - focus mobile optimization tests",
                "Test different value propositions for different traffic sources",
                "Investigate device-specific user experience improvements",
                "Consider personalization based on user segments"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete A/B testing analysis"""
        print("🚀 Starting A/B Testing Analysis...")
        
        # Generate and analyze data
        self.generate_ab_test_data()
        self.perform_statistical_tests()
        
        # Create visualizations
        dashboard = self.create_ab_testing_dashboard()
        
        # Generate recommendations
        recommendations = self.generate_testing_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        dashboard.write_html('results/ab_testing_dashboard.html')
        self.data.to_csv('results/ab_test_data.csv', index=False)
        
        # Save test results
        results_df = pd.DataFrame(self.test_results).T
        results_df.to_csv('results/statistical_test_results.csv')
        
        import json
        with open('results/ab_testing_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        significant_tests = sum(1 for result in self.test_results.values() if result['is_significant'])
        avg_improvement = np.mean([r['relative_improvement'] for r in self.test_results.values()])
        
        summary_report = {
            'dataset_summary': {
                'total_users': len(self.data),
                'total_tests': len(self.test_results),
                'date_range': f"{self.data['test_date'].min()} to {self.data['test_date'].max()}",
                'overall_conversion_rate': f"{self.data['converted'].mean():.2%}"
            },
            'statistical_results': {
                'significant_tests': f"{significant_tests}/{len(self.test_results)}",
                'average_improvement': f"{avg_improvement:.2%}",
                'average_p_value': f"{np.mean([r['p_value'] for r in self.test_results.values()]):.4f}",
                'recommended_sample_size': f"{self.calculate_sample_size_recommendations():,} per variant"
            },
            'key_insights': [
                f"Best performing test: {max(self.test_results.items(), key=lambda x: x[1]['relative_improvement'])[0]}",
                f"{significant_tests} out of {len(self.test_results)} tests showed statistical significance",
                f"Average relative improvement: {avg_improvement:.2%}",
                f"Mobile users have {self.data[self.data['device_type']=='Mobile']['converted'].mean():.2%} conversion rate"
            ]
        }
        
        with open('results/ab_testing_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print("\n✅ A/B Testing Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/ab_testing_dashboard.html")
        print("   - results/ab_test_data.csv")
        print("   - results/statistical_test_results.csv")
        print("   - results/ab_testing_recommendations.json")
        print("   - results/ab_testing_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Users Analyzed: {len(self.data):,}")
        print(f"   - Tests Conducted: {len(self.test_results)}")
        print(f"   - Significant Results: {significant_tests}")
        print(f"   - Average Improvement: {avg_improvement:.2%}")
        
        return summary_report

if __name__ == "__main__":
    ab_system = ABTestingSystem()
    results = ab_system.run_complete_analysis()