"""
Market Basket Analysis System (Fixed)
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
            all_products.extend(items)
        
        transactions = []
        
        for i in range(n_transactions):
            # Transaction size (1-8 items)
            transaction_size = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 
                                              p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
            
            # Select products with some correlation patterns
            selected_products = set()
            
            # Add some common associations
            if np.random.random() < 0.3:  # Milk + Bread
                if len(selected_products) < transaction_size - 1:
                    selected_products.update(['Milk', 'Bread'])
            
            # Fill remaining slots randomly
            while len(selected_products) < transaction_size:
                product = np.random.choice(all_products)
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
        
        return self.data
    
    def find_frequent_itemsets(self, min_support=0.01):
        """Find frequent itemsets using simplified approach"""
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
        
        total_frequent = sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
        print(f"✅ Found {total_frequent} frequent itemsets")
    
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
        
        # Sort by lift (descending)
        self.association_rules.sort(key=lambda x: x['lift'], reverse=True)
        
        print(f"✅ Generated {len(self.association_rules)} association rules")
    
    def run_complete_analysis(self):
        """Run complete market basket analysis"""
        print("🚀 Starting Market Basket Analysis...")
        
        # Generate and analyze data
        self.generate_transaction_data()
        self.find_frequent_itemsets()
        self.generate_association_rules()
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        
        self.data.to_csv('results/transaction_data.csv', index=False)
        
        # Save association rules
        rules_df = pd.DataFrame(self.association_rules)
        rules_df.to_csv('results/association_rules.csv', index=False)
        
        print("\n✅ Market Basket Analysis Complete!")
        print("📁 Results saved:")
        print("   - results/transaction_data.csv")
        print("   - results/association_rules.csv")
        
        return {'status': 'complete', 'rules': len(self.association_rules)}

if __name__ == "__main__":
    basket_system = MarketBasketAnalysisSystem()
    results = basket_system.run_complete_analysis()