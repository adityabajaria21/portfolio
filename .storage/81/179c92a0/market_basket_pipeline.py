"""
Market Basket Analysis System
Association rule mining for product recommendations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class MarketBasketAnalysisSystem:
    def __init__(self):
        self.frequent_itemsets = {}
        self.association_rules = []
        
    def generate_transaction_data(self, n_transactions=50000):
        """Generate realistic transaction dataset"""
        np.random.seed(42)
        
        # Product catalog
        products = {
            'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch', 'Camera'],
            'Groceries': ['Milk', 'Bread', 'Eggs', 'Cheese', 'Apples', 'Bananas', 'Chicken', 'Rice'],
            'Clothing': ['T-Shirt', 'Jeans', 'Sneakers', 'Jacket', 'Dress', 'Shorts'],
            'Home': ['Towels', 'Sheets', 'Pillow', 'Blanket', 'Curtains', 'Lamp'],
            'Beauty': ['Shampoo', 'Conditioner', 'Soap', 'Lotion', 'Perfume', 'Makeup']
        }
        
        all_products = []
        for category, items in products.items():
            all_products.extend([(item, category) for item in items])
        
        transactions = []
        
        for i in range(n_transactions):
            # Transaction size (1-8 items)
            transaction_size = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 
                                              p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
            
            # Select products with some correlation patterns
            selected_products = set()
            
            # Add some common associations
            if np.random.random() < 0.3:  # Milk + Bread
                if 'Milk' not in selected_products and 'Bread' not in selected_products:
                    selected_products.update(['Milk', 'Bread'])
            
            if np.random.random() < 0.2:  # Electronics bundle
                electronics = [p[0] for p in all_products if p[1] == 'Electronics']
                selected_products.add(np.random.choice(electronics))
                if np.random.random() < 0.4:
                    selected_products.add(np.random.choice(electronics))
            
            # Fill remaining slots randomly
            while len(selected_products) < transaction_size:
                product, category = np.random.choice(all_products)
                selected_products.add(product)
            
            # Convert to list and trim if necessary
            selected_products = list(selected_products)[:transaction_size]
            
            transaction = {
                'transaction_id': f'T{i+1:06d}',
                'customer_id': f'C{np.random.randint(1, 10000):05d}',
                'transaction_date': pd.date_range(start='2023-01-01', end='2024-01-01', periods=n_transactions)[i],
                'products': selected_products,
                'total_items': len(selected_products),
                'total_amount': len(selected_products) * np.random.uniform(10, 50)
            }
            
            transactions.append(transaction)
        
        self.data = pd.DataFrame(transactions)
        
        # Create binary matrix for association analysis
        all_unique_products = set()
        for products in self.data['products']:
            all_unique_products.update(products)
        
        self.all_products = sorted(list(all_unique_products))
        
        # Create binary matrix
        binary_matrix = []
        for products in self.data['products']:
            row = [1 if product in products else 0 for product in self.all_products]
            binary_matrix.append(row)
        
        self.binary_df = pd.DataFrame(binary_matrix, columns=self.all_products)
        
        print(f"✅ Generated {len(self.data):,} transactions")
        print(f"   - Unique products: {len(self.all_products)}")
        print(f"   - Average items per transaction: {self.data['total_items'].mean():.2f}")
        print(f"   - Date range: {self.data['transaction_date'].min()} to {self.data['transaction_date'].max()}")
        
        return self.data
    
    def find_frequent_itemsets(self, min_support=0.01):
        """Find frequent itemsets using Apriori-like approach"""
        print("🔄 Finding frequent itemsets...")
        
        n_transactions = len(self.binary_df)
        min_support_count = int(min_support * n_transactions)
        
        # Find frequent 1-itemsets
        frequent_1_itemsets = {}
        for product in self.all_products:
            support_count = self.binary_df[product].sum()
            if support_count >= min_support_count:
                frequent_1_itemsets[frozenset([product])] = support_count / n_transactions
        
        self.frequent_itemsets[1] = frequent_1_itemsets
        
        # Find frequent 2-itemsets
        frequent_2_itemsets = {}
        for product1, product2 in combinations(self.all_products, 2):
            support_count = (self.binary_df[product1] & self.binary_df[product2]).sum()
            if support_count >= min_support_count:
                frequent_2_itemsets[frozenset([product1, product2])] = support_count / n_transactions
        
        self.frequent_itemsets[2] = frequent_2_itemsets
        
        # Find frequent 3-itemsets
        frequent_3_itemsets = {}
        for product1, product2, product3 in combinations(self.all_products, 3):
            support_count = (self.binary_df[product1] & self.binary_df[product2] & self.binary_df[product3]).sum()
            if support_count >= min_support_count:
                frequent_3_itemsets[frozenset([product1, product2, product3])] = support_count / n_transactions
        
        self.frequent_itemsets[3] = frequent_3_itemsets
        
        total_frequent = sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
        print(f"✅ Found {total_frequent} frequent itemsets")
        print(f"   - 1-itemsets: {len(frequent_1_itemsets)}")
        print(f"   - 2-itemsets: {len(frequent_2_itemsets)}")
        print(f"   - 3-itemsets: {len(frequent_3_itemsets)}")
    
    def generate_association_rules(self, min_confidence=0.5):
        """Generate association rules from frequent itemsets"""
        print("🔄 Generating association rules...")
        
        self.association_rules = []
        
        # Generate rules from 2-itemsets
        for itemset, support in self.frequent_itemsets[2].items():
            items = list(itemset)
            
            # Rule: A -> B
            antecedent = frozenset([items[0]])
            consequent = frozenset([items[1]])
            
            if antecedent in self.frequent_itemsets[1]:
                confidence = support / self.frequent_itemsets[1][antecedent]
                if confidence >= min_confidence:
                    lift = confidence / self.frequent_itemsets[1][consequent]
                    
                    self.association_rules.append({
                        'antecedent': list(antecedent),
                        'consequent': list(consequent),
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
            
            # Rule: B -> A
            antecedent = frozenset([items[1]])
            consequent = frozenset([items[0]])
            
            if antecedent in self.frequent_itemsets[1]:
                confidence = support / self.frequent_itemsets[1][antecedent]
                if confidence >= min_confidence:
                    lift = confidence / self.frequent_itemsets[1][consequent]
                    
                    self.association_rules.append({
                        'antecedent': list(antecedent),
                        'consequent': list(consequent),
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
        
        # Generate rules from 3-itemsets
        for itemset, support in self.frequent_itemsets[3].items():
            items = list(itemset)
            
            # Generate all possible 2->1 rules
            for i in range(len(items)):
                consequent = frozenset([items[i]])
                antecedent = frozenset([item for j, item in enumerate(items) if j != i])
                
                if antecedent in self.frequent_itemsets[2]:
                    confidence = support / self.frequent_itemsets[2][antecedent]
                    if confidence >= min_confidence:
                        lift = confidence / self.frequent_itemsets[1][consequent]
                        
                        self.association_rules.append({
                            'antecedent': list(antecedent),
                            'consequent': list(consequent),
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })
        
        # Sort by lift (descending)
        self.association_rules.sort(key=lambda x: x['lift'], reverse=True)
        
        print(f"✅ Generated {len(self.association_rules)} association rules")
        print(f"   - Average confidence: {np.mean([rule['confidence'] for rule in self.association_rules]):.3f}")
        print(f"   - Average lift: {np.mean([rule['lift'] for rule in self.association_rules]):.3f}")
    
    def create_market_basket_dashboard(self):
        """Create comprehensive market basket analysis dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Top Products by Frequency', 'Transaction Size Distribution', 'Monthly Transaction Trends',
                          'Top Association Rules (by Lift)', 'Support vs Confidence', 'Product Co-occurrence Matrix',
                          'Customer Purchase Patterns', 'Revenue by Product', 'Cross-selling Opportunities'),
            specs=[[{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Top Products by Frequency
        product_frequency = {}
        for products in self.data['products']:
            for product in products:
                product_frequency[product] = product_frequency.get(product, 0) + 1
        
        top_products = sorted(product_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        fig.add_trace(
            go.Bar(x=[p[0] for p in top_products], y=[p[1] for p in top_products], name='Product Frequency'),
            row=1, col=1
        )
        
        # 2. Transaction Size Distribution
        fig.add_trace(
            go.Histogram(x=self.data['total_items'], nbinsx=10, name='Transaction Size'),
            row=1, col=2
        )
        
        # 3. Monthly Transaction Trends
        monthly_transactions = self.data.groupby(self.data['transaction_date'].dt.to_period('M')).size()
        fig.add_trace(
            go.Bar(x=[str(x) for x in monthly_transactions.index], y=monthly_transactions.values, name='Monthly Transactions'),
            row=1, col=3
        )
        
        # 4. Top Association Rules
        if self.association_rules:
            top_rules = self.association_rules[:10]
            rule_labels = [f"{' + '.join(rule['antecedent'])} → {' + '.join(rule['consequent'])}" for rule in top_rules]
            fig.add_trace(
                go.Bar(x=[rule['lift'] for rule in top_rules], y=rule_labels, orientation='h', name='Lift'),
                row=2, col=1
            )
        
        # 5. Support vs Confidence Scatter
        if self.association_rules:
            fig.add_trace(
                go.Scatter(x=[rule['support'] for rule in self.association_rules], 
                          y=[rule['confidence'] for rule in self.association_rules],
                          mode='markers', name='Rules', opacity=0.6),
                row=2, col=2
            )
        
        # 6. Product Co-occurrence Matrix (simplified)
        top_10_products = [p[0] for p in top_products]
        cooccurrence_matrix = np.zeros((len(top_10_products), len(top_10_products)))
        
        for products in self.data['products']:
            for i, product1 in enumerate(top_10_products):
                for j, product2 in enumerate(top_10_products):
                    if product1 in products and product2 in products and i != j:
                        cooccurrence_matrix[i][j] += 1
        
        fig.add_trace(
            go.Heatmap(z=cooccurrence_matrix, x=top_10_products, y=top_10_products, name='Co-occurrence'),
            row=2, col=3
        )
        
        # 7. Customer Purchase Patterns
        customer_stats = self.data.groupby('customer_id').agg({
            'total_items': 'mean',
            'total_amount': 'mean'
        }).reset_index()
        
        avg_items_per_customer = customer_stats['total_items'].mean()
        fig.add_trace(
            go.Bar(x=['Avg Items per Customer'], y=[avg_items_per_customer], name='Customer Patterns'),
            row=3, col=1
        )
        
        # 8. Revenue by Product
        product_revenue = {}
        for idx, row in self.data.iterrows():
            revenue_per_item = row['total_amount'] / row['total_items']
            for product in row['products']:
                product_revenue[product] = product_revenue.get(product, 0) + revenue_per_item
        
        top_revenue_products = sorted(product_revenue.items(), key=lambda x: x[1], reverse=True)[:10]
        fig.add_trace(
            go.Bar(x=[p[0] for p in top_revenue_products], y=[p[1] for p in top_revenue_products], name='Revenue'),
            row=3, col=2
        )
        
        # 9. Cross-selling Opportunities Table
        if self.association_rules:
            top_cross_sell = self.association_rules[:5]
            fig.add_trace(
                go.Table(
                    header=dict(values=['If Customer Buys', 'Recommend', 'Confidence', 'Lift']),
                    cells=dict(values=[
                        [' + '.join(rule['antecedent']) for rule in top_cross_sell],
                        [' + '.join(rule['consequent']) for rule in top_cross_sell],
                        [f"{rule['confidence']:.2%}" for rule in top_cross_sell],
                        [f"{rule['lift']:.2f}" for rule in top_cross_sell]
                    ])
                ),
                row=3, col=3
            )
        
        fig.update_layout(height=1200, title_text="Market Basket Analysis Dashboard")
        return fig
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        # Calculate key metrics
        avg_transaction_size = self.data['total_items'].mean()
        top_products = {}
        for products in self.data['products']:
            for product in products:
                top_products[product] = top_products.get(product, 0) + 1
        
        most_popular = max(top_products.items(), key=lambda x: x[1])
        
        recommendations = {
            'cross_selling_opportunities': [
                f"Top cross-selling rule: {self.association_rules[0]['antecedent']} → {self.association_rules[0]['consequent']}" if self.association_rules else "Generate more association rules",
                f"Bundle products with high lift values (>1.5) for promotions",
                f"Place frequently bought together items near each other in store",
                "Implement 'customers who bought this also bought' recommendations"
            ],
            'inventory_management': [
                f"Stock {most_popular[0]} heavily - it's the most popular product",
                f"Average transaction size is {avg_transaction_size:.1f} items - optimize basket size",
                "Monitor seasonal patterns in product combinations",
                "Adjust inventory based on association rule strength"
            ],
            'marketing_strategies': [
                "Create product bundles based on strong association rules",
                "Target customers with personalized recommendations",
                "Design promotional campaigns around frequent itemsets",
                "Use lift values to determine discount strategies"
            ],
            'store_layout': [
                "Position complementary products close together",
                "Create themed sections based on product associations",
                "Use association rules to optimize product placement",
                "Design checkout area with high-lift impulse items"
            ]
        }
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete market basket analysis"""
        print("🚀 Starting Market Basket Analysis...")
        
        # Generate and analyze data
        self.generate_transaction_data()
        self.find_frequent_itemsets()
        self.generate_association_rules()
        
        # Create visualizations
        dashboard = self.create_market_basket_dashboard()
        
        # Generate recommendations
        recommendations = self.generate_business_recommendations()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        dashboard.write_html('results/market_basket_dashboard.html')
        self.data.to_csv('results/transaction_data.csv', index=False)
        
        # Save association rules
        rules_df = pd.DataFrame(self.association_rules)
        rules_df.to_csv('results/association_rules.csv', index=False)
        
        import json
        with open('results/market_basket_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'dataset_summary': {
                'total_transactions': len(self.data),
                'unique_products': len(self.all_products),
                'average_transaction_size': f"{self.data['total_items'].mean():.2f}",
                'total_revenue': f"${self.data['total_amount'].sum():,.2f}"
            },
            'analysis_results': {
                'frequent_itemsets_found': sum(len(itemsets) for itemsets in self.frequent_itemsets.values()),
                'association_rules_generated': len(self.association_rules),
                'average_confidence': f"{np.mean([rule['confidence'] for rule in self.association_rules]):.3f}" if self.association_rules else "N/A",
                'average_lift': f"{np.mean([rule['lift'] for rule in self.association_rules]):.3f}" if self.association_rules else "N/A"
            },
            'top_insights': [
                f"Most popular product: {max({p: sum(1 for products in self.data['products'] if p in products) for p in self.all_products}.items(), key=lambda x: x[1])[0]}",
                f"Best cross-selling opportunity: {self.association_rules[0]['antecedent']} → {self.association_rules[0]['consequent']}" if self.association_rules else "Generate more rules",
                f"Average items per transaction: {self.data['total_items'].mean():.2f}"
            ]
        }
        
        with open('results/market_basket_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("\n✅ Market Basket Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/market_basket_dashboard.html")
        print("   - results/transaction_data.csv")
        print("   - results/association_rules.csv")
        print("   - results/market_basket_recommendations.json")
        print("   - results/market_basket_summary.json")
        
        print(f"\n📊 Key Results:")
        print(f"   - Transactions Analyzed: {len(self.data):,}")
        print(f"   - Association Rules: {len(self.association_rules)}")
        print(f"   - Unique Products: {len(self.all_products)}")
        print(f"   - Average Transaction Size: {self.data['total_items'].mean():.2f}")
        
        return summary_report

if __name__ == "__main__":
    basket_system = MarketBasketAnalysisSystem()
    results = basket_system.run_complete_analysis()