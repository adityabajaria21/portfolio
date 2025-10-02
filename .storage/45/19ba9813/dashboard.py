"""
Interactive Dashboard for E-commerce CLV Analysis
Creates a comprehensive Plotly Dash dashboard
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class CLVDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.df_transactions = None
        self.df_rfm = None
        self.df_clusters = None
        self.df_clv = None
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """Load all processed data files"""
        try:
            self.df_transactions = pd.read_csv('data/processed/cleaned_data.csv')
            self.df_rfm = pd.read_csv('data/processed/customer_segments.csv')
            self.df_clusters = pd.read_csv('data/processed/clustered_customers.csv')
            self.df_clv = pd.read_csv('data/processed/clv_predictions.csv')
            print("All data files loaded successfully for dashboard")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            # Create sample data if files don't exist
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for dashboard demonstration"""
        np.random.seed(42)
        n_customers = 1000
        
        # Sample RFM data
        self.df_rfm = pd.DataFrame({
            'CustomerID': range(10000, 10000 + n_customers),
            'Recency': np.random.exponential(50, n_customers),
            'Frequency': np.random.poisson(5, n_customers) + 1,
            'Monetary': np.random.lognormal(4, 1, n_customers),
            'R_Score': np.random.randint(1, 6, n_customers),
            'F_Score': np.random.randint(1, 6, n_customers),
            'M_Score': np.random.randint(1, 6, n_customers),
            'RFM_Score_Weighted': np.random.uniform(1, 5, n_customers),
            'Segment': np.random.choice(['Champions', 'Loyal Customers', 'New Customers', 'At Risk', 'Lost'], n_customers)
        })
        
        # Sample CLV data
        self.df_clv = pd.DataFrame({
            'CustomerID': range(10000, 10000 + n_customers),
            'frequency': np.random.poisson(3, n_customers),
            'recency': np.random.exponential(30, n_customers),
            'T': np.random.uniform(30, 365, n_customers),
            'monetary_value': np.random.lognormal(3, 0.5, n_customers),
            'predicted_clv': np.random.lognormal(5, 1, n_customers),
            'prob_alive': np.random.beta(2, 1, n_customers),
            'clv_segment': np.random.choice(['Low Value', 'Below Average', 'Average', 'Above Average', 'High Value'], n_customers)
        })
        
        # Sample cluster data
        self.df_clusters = self.df_rfm.copy()
        self.df_clusters['Cluster'] = np.random.randint(0, 5, n_customers)
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("E-commerce Customer Analytics Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
            
            # Summary Cards
            html.Div([
                html.Div([
                    html.H3(f"{len(self.df_clv):,}", style={'margin': 0, 'color': '#3498db'}),
                    html.P("Total Customers", style={'margin': 0})
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10, 'margin': 10}),
                
                html.Div([
                    html.H3(f"${self.df_clv['predicted_clv'].sum():,.0f}", style={'margin': 0, 'color': '#e74c3c'}),
                    html.P("Total Predicted CLV", style={'margin': 0})
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10, 'margin': 10}),
                
                html.Div([
                    html.H3(f"${self.df_clv['predicted_clv'].mean():.0f}", style={'margin': 0, 'color': '#f39c12'}),
                    html.P("Average CLV", style={'margin': 0})
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10, 'margin': 10}),
                
                html.Div([
                    html.H3(f"{self.df_clv['prob_alive'].mean():.1%}", style={'margin': 0, 'color': '#27ae60'}),
                    html.P("Avg Probability Alive", style={'margin': 0})
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10, 'margin': 10})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
            
            # Tabs
            dcc.Tabs(id="tabs", value='rfm-tab', children=[
                dcc.Tab(label='RFM Analysis', value='rfm-tab'),
                dcc.Tab(label='Customer Clustering', value='clustering-tab'),
                dcc.Tab(label='CLV Prediction', value='clv-tab'),
                dcc.Tab(label='Business Insights', value='insights-tab')
            ]),
            
            html.Div(id='tab-content')
        ])
    
    def create_rfm_tab(self):
        """Create RFM analysis tab content"""
        return html.Div([
            html.H2("RFM Analysis", style={'textAlign': 'center', 'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.histogram(
                            self.df_rfm, x='Recency', nbins=50,
                            title='Recency Distribution',
                            labels={'Recency': 'Days Since Last Purchase'}
                        )
                    )
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.histogram(
                            self.df_rfm, x='Frequency', nbins=30,
                            title='Frequency Distribution',
                            labels={'Frequency': 'Number of Purchases'}
                        )
                    )
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.histogram(
                            self.df_rfm, x='Monetary', nbins=50,
                            title='Monetary Distribution',
                            labels={'Monetary': 'Total Spent ($)'}
                        )
                    )
                ], style={'width': '33%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.pie(
                            self.df_rfm.groupby('Segment').size().reset_index(name='count'),
                            values='count', names='Segment',
                            title='Customer Segments Distribution'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.scatter(
                            self.df_rfm, x='Frequency', y='Monetary', color='Segment',
                            title='Frequency vs Monetary by Segment',
                            hover_data=['CustomerID', 'Recency']
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def create_clustering_tab(self):
        """Create clustering analysis tab content"""
        return html.Div([
            html.H2("Customer Clustering", style={'textAlign': 'center', 'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.scatter_3d(
                            self.df_clusters, x='Recency', y='Frequency', z='Monetary',
                            color='Cluster', title='3D RFM Clusters',
                            hover_data=['CustomerID']
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.bar(
                            self.df_clusters.groupby('Cluster').size().reset_index(name='count'),
                            x='Cluster', y='count',
                            title='Cluster Size Distribution'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.scatter(
                            self.df_clusters, x='Recency', y='Frequency', color='Cluster',
                            title='Recency vs Frequency by Cluster',
                            hover_data=['CustomerID', 'Monetary']
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.box(
                            self.df_clusters, x='Cluster', y='Monetary',
                            title='Monetary Value Distribution by Cluster'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def create_clv_tab(self):
        """Create CLV analysis tab content"""
        return html.Div([
            html.H2("Customer Lifetime Value Prediction", style={'textAlign': 'center', 'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.histogram(
                            self.df_clv, x='predicted_clv', nbins=50,
                            title='CLV Distribution',
                            labels={'predicted_clv': 'Predicted CLV ($)'}
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.pie(
                            self.df_clv.groupby('clv_segment').size().reset_index(name='count'),
                            values='count', names='clv_segment',
                            title='CLV Segments Distribution'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.scatter(
                            self.df_clv, x='frequency', y='predicted_clv', color='clv_segment',
                            title='CLV vs Purchase Frequency',
                            hover_data=['CustomerID', 'prob_alive']
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.scatter(
                            self.df_clv, x='prob_alive', y='predicted_clv', color='clv_segment',
                            title='CLV vs Probability Alive',
                            hover_data=['CustomerID', 'frequency']
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Top customers table
            html.H3("Top 20 Customers by CLV", style={'textAlign': 'center', 'marginTop': 30}),
            html.Div([
                dash_table.DataTable(
                    data=self.df_clv.nlargest(20, 'predicted_clv')[['CustomerID', 'predicted_clv', 'prob_alive', 'clv_segment']].round(2).to_dict('records'),
                    columns=[
                        {'name': 'Customer ID', 'id': 'CustomerID'},
                        {'name': 'Predicted CLV ($)', 'id': 'predicted_clv', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Probability Alive', 'id': 'prob_alive', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'CLV Segment', 'id': 'clv_segment'}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{clv_segment} = High Value'},
                            'backgroundColor': '#e8f5e8',
                            'color': 'black',
                        }
                    ]
                )
            ])
        ])
    
    def create_insights_tab(self):
        """Create business insights tab content"""
        # Calculate key metrics
        total_clv = self.df_clv['predicted_clv'].sum()
        avg_clv = self.df_clv['predicted_clv'].mean()
        
        # Segment analysis
        segment_stats = self.df_clv.groupby('clv_segment').agg({
            'CustomerID': 'count',
            'predicted_clv': ['sum', 'mean'],
            'prob_alive': 'mean'
        }).round(2)
        
        segment_stats.columns = ['customer_count', 'total_clv', 'avg_clv', 'avg_prob_alive']
        segment_stats['clv_percentage'] = (segment_stats['total_clv'] / total_clv * 100).round(1)
        segment_stats = segment_stats.reset_index()
        
        # At-risk high-value customers
        at_risk_high_value = self.df_clv[
            (self.df_clv['prob_alive'] < 0.3) & 
            (self.df_clv['predicted_clv'] > self.df_clv['predicted_clv'].quantile(0.8))
        ]
        
        return html.Div([
            html.H2("Business Insights & Recommendations", style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Key Metrics
            html.Div([
                html.H3("Key Business Metrics", style={'color': '#2c3e50'}),
                html.Ul([
                    html.Li(f"Total Customer Base: {len(self.df_clv):,} customers"),
                    html.Li(f"Total Predicted CLV (1 Year): ${total_clv:,.0f}"),
                    html.Li(f"Average CLV per Customer: ${avg_clv:.0f}"),
                    html.Li(f"Customers at Risk of Churn: {len(self.df_clv[self.df_clv['prob_alive'] < 0.3]):,}"),
                    html.Li(f"High-Value At-Risk Customers: {len(at_risk_high_value):,}")
                ], style={'fontSize': 16, 'lineHeight': 2})
            ], style={'marginBottom': 30}),
            
            # Segment Performance
            html.Div([
                html.H3("Segment Performance", style={'color': '#2c3e50'}),
                dash_table.DataTable(
                    data=segment_stats.to_dict('records'),
                    columns=[
                        {'name': 'CLV Segment', 'id': 'clv_segment'},
                        {'name': 'Customer Count', 'id': 'customer_count'},
                        {'name': 'Total CLV ($)', 'id': 'total_clv', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Avg CLV ($)', 'id': 'avg_clv', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Avg Prob Alive', 'id': 'avg_prob_alive', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                        {'name': 'CLV %', 'id': 'clv_percentage', 'type': 'numeric', 'format': {'specifier': '.1f'}}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{clv_segment} = High Value'},
                            'backgroundColor': '#d5f4e6',
                            'color': 'black',
                        }
                    ]
                )
            ], style={'marginBottom': 30}),
            
            # Strategic Recommendations
            html.Div([
                html.H3("Strategic Recommendations", style={'color': '#2c3e50'}),
                html.Div([
                    html.Div([
                        html.H4("🎯 High Value Customers", style={'color': '#27ae60'}),
                        html.P("Focus on retention and upselling. These customers represent the highest ROI."),
                        html.Ul([
                            html.Li("Implement VIP customer service"),
                            html.Li("Offer exclusive products and early access"),
                            html.Li("Create personalized experiences")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.H4("⚠️ At-Risk Customers", style={'color': '#e74c3c'}),
                        html.P(f"{len(at_risk_high_value)} high-value customers are at risk of churning."),
                        html.Ul([
                            html.Li("Launch immediate re-engagement campaigns"),
                            html.Li("Offer personalized discounts"),
                            html.Li("Conduct customer satisfaction surveys")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ]),
                
                html.Div([
                    html.Div([
                        html.H4("📈 Growth Opportunities", style={'color': '#3498db'}),
                        html.P("Focus acquisition efforts on customers similar to high-value segments."),
                        html.Ul([
                            html.Li(f"Target new customers with potential CLV > ${avg_clv:.0f}"),
                            html.Li("Develop lookalike audiences based on Champions segment"),
                            html.Li("Optimize marketing spend allocation")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                    
                    html.Div([
                        html.H4("💡 Operational Insights", style={'color': '#f39c12'}),
                        html.P("Optimize operations based on customer behavior patterns."),
                        html.Ul([
                            html.Li("Adjust inventory based on high-value customer preferences"),
                            html.Li("Optimize customer service resources"),
                            html.Li("Implement predictive analytics for proactive support")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginTop': 20})
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'value')
        )
        def render_content(tab):
            if tab == 'rfm-tab':
                return self.create_rfm_tab()
            elif tab == 'clustering-tab':
                return self.create_clustering_tab()
            elif tab == 'clv-tab':
                return self.create_clv_tab()
            elif tab == 'insights-tab':
                return self.create_insights_tab()
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        print(f"Starting dashboard server on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = CLVDashboard()
    dashboard.run_server()