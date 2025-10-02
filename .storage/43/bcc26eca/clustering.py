"""
Customer Clustering Module
Implements K-Means clustering for customer segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class CustomerClustering:
    def __init__(self, data_path='data/processed/rfm_analysis.csv'):
        self.data_path = data_path
        self.df = None
        self.scaled_features = None
        self.scaler = None
        self.kmeans_model = None
        self.optimal_k = None
        
    def load_data(self):
        """Load RFM data for clustering"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"RFM data loaded: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("RFM data not found. Please run RFM analysis first.")
            return None
    
    def prepare_features(self, features=['Recency', 'Frequency', 'Monetary']):
        """Prepare and scale features for clustering"""
        if self.df is None:
            self.load_data()
        
        # Select features for clustering
        feature_data = self.df[features].copy()
        
        # Handle any missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        # Scale the features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(feature_data)
        
        print(f"Features prepared and scaled: {self.scaled_features.shape}")
        print(f"Features used: {features}")
        
        return self.scaled_features
    
    def find_optimal_clusters(self, max_k=10, methods=['elbow', 'silhouette']):
        """Find optimal number of clusters using multiple methods"""
        if self.scaled_features is None:
            self.prepare_features()
        
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        print("Finding optimal number of clusters...")
        
        for k in k_range:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.scaled_features, cluster_labels))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].grid(True)
        
        # Silhouette analysis
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True)
        
        # Calinski-Harabasz Index
        axes[2].plot(k_range, calinski_scores, 'go-')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k based on silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Calculate elbow point (simplified method)
        # Find the point with maximum distance from line connecting first and last points
        def calculate_elbow_point(inertias):
            n_points = len(inertias)
            all_coord = np.vstack((range(n_points), inertias)).T
            first_point = all_coord[0]
            last_point = all_coord[-1]
            
            # Calculate distance from each point to the line
            distances = []
            for coord in all_coord:
                distances.append(np.abs(np.cross(last_point - first_point, first_point - coord)) / 
                               np.linalg.norm(last_point - first_point))
            
            return np.argmax(distances) + 2  # +2 because we start from k=2
        
        optimal_k_elbow = calculate_elbow_point(inertias)
        
        print(f"\nOptimal k suggestions:")
        print(f"Silhouette method: {optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
        print(f"Elbow method: {optimal_k_elbow}")
        
        # Use silhouette method as primary recommendation
        self.optimal_k = optimal_k_silhouette
        
        return {
            'k_range': k_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_elbow': optimal_k_elbow
        }
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_k
        
        if self.scaled_features is None:
            self.prepare_features()
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)
        
        # Add cluster labels to the dataframe
        self.df['Cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = self.df.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'std'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.rename(columns={'CustomerID_count': 'Customer_Count'})
        
        # Calculate cluster percentages
        cluster_stats['Percentage'] = (cluster_stats['Customer_Count'] / len(self.df) * 100).round(2)
        
        print(f"\nK-means clustering completed with {n_clusters} clusters")
        print(f"Silhouette score: {silhouette_score(self.scaled_features, cluster_labels):.3f}")
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        # Save results
        self.df.to_csv('data/processed/clustered_customers.csv', index=False)
        cluster_stats.to_csv('data/processed/cluster_statistics.csv')
        
        return self.df, cluster_stats
    
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        if 'Cluster' not in self.df.columns:
            self.perform_clustering()
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Customer Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Color palette for clusters
        n_clusters = self.df['Cluster'].nunique()
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # 1. 3D scatter plot of RFM
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                      cluster_data['Monetary'], c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('3D RFM Clusters')
        ax.legend()
        
        # Remove the 3D plot from the subplot grid
        axes[0, 0].remove()
        
        # 2. Recency vs Frequency
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            axes[0, 1].scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        axes[0, 1].set_xlabel('Recency (days)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Recency vs Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Frequency vs Monetary
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            axes[0, 2].scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        axes[0, 2].set_xlabel('Frequency')
        axes[0, 2].set_ylabel('Monetary ($)')
        axes[0, 2].set_title('Frequency vs Monetary')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Cluster size distribution
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        bars = axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color=colors)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xticks(cluster_counts.index)
        
        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                           str(count), ha='center', va='bottom')
        
        # 5. PCA visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(self.scaled_features)
        
        for i, cluster in enumerate(sorted(self.df['Cluster'].unique())):
            cluster_mask = self.df['Cluster'] == cluster
            axes[1, 1].scatter(pca_features[cluster_mask, 0], pca_features[cluster_mask, 1], 
                             c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1, 1].set_title('PCA Visualization of Clusters')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Cluster characteristics heatmap
        cluster_means = self.df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # Normalize for better visualization
        cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        im = axes[1, 2].imshow(cluster_means_normalized.T, cmap='RdYlBu_r', aspect='auto')
        axes[1, 2].set_xticks(range(len(cluster_means_normalized)))
        axes[1, 2].set_xticklabels([f'Cluster {i}' for i in cluster_means_normalized.index])
        axes[1, 2].set_yticks(range(len(cluster_means_normalized.columns)))
        axes[1, 2].set_yticklabels(cluster_means_normalized.columns)
        axes[1, 2].set_title('Cluster Characteristics\n(Normalized)')
        
        # Add text annotations
        for i in range(len(cluster_means_normalized)):
            for j in range(len(cluster_means_normalized.columns)):
                text = axes[1, 2].text(i, j, f'{cluster_means_normalized.iloc[i, j]:.1f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 2])
        plt.tight_layout()
        plt.savefig('visualizations/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def interpret_clusters(self):
        """Provide interpretation and business insights for each cluster"""
        if 'Cluster' not in self.df.columns:
            self.perform_clustering()
        
        cluster_profiles = {}
        
        for cluster in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            
            avg_recency = cluster_data['Recency'].mean()
            avg_frequency = cluster_data['Frequency'].mean()
            avg_monetary = cluster_data['Monetary'].mean()
            cluster_size = len(cluster_data)
            
            # Determine cluster characteristics
            if avg_recency <= self.df['Recency'].quantile(0.33):
                recency_level = "Recent"
            elif avg_recency <= self.df['Recency'].quantile(0.67):
                recency_level = "Moderate"
            else:
                recency_level = "Long ago"
            
            if avg_frequency >= self.df['Frequency'].quantile(0.67):
                frequency_level = "High"
            elif avg_frequency >= self.df['Frequency'].quantile(0.33):
                frequency_level = "Medium"
            else:
                frequency_level = "Low"
            
            if avg_monetary >= self.df['Monetary'].quantile(0.67):
                monetary_level = "High"
            elif avg_monetary >= self.df['Monetary'].quantile(0.33):
                monetary_level = "Medium"
            else:
                monetary_level = "Low"
            
            # Generate cluster name and strategy
            cluster_name = f"{recency_level}_{frequency_level}_{monetary_level}"
            
            # Business interpretation
            if recency_level == "Recent" and frequency_level == "High" and monetary_level == "High":
                interpretation = "VIP Customers"
                strategy = "Maintain satisfaction with premium service and exclusive offers"
            elif recency_level == "Recent" and frequency_level == "High":
                interpretation = "Loyal Customers"
                strategy = "Upsell and cross-sell opportunities, loyalty programs"
            elif recency_level == "Recent" and monetary_level == "High":
                interpretation = "Big Spenders"
                strategy = "Encourage repeat purchases with personalized recommendations"
            elif recency_level == "Recent":
                interpretation = "New/Active Customers"
                strategy = "Nurture relationship, encourage engagement"
            elif frequency_level == "High" and monetary_level == "High":
                interpretation = "At Risk VIPs"
                strategy = "Immediate re-engagement campaigns, personal outreach"
            elif frequency_level == "High":
                interpretation = "At Risk Loyal"
                strategy = "Win-back campaigns, special offers"
            elif monetary_level == "High":
                interpretation = "At Risk Big Spenders"
                strategy = "Premium win-back offers, account management"
            else:
                interpretation = "Lost/Inactive"
                strategy = "Low-cost reactivation or customer acquisition focus"
            
            cluster_profiles[cluster] = {
                'name': cluster_name,
                'interpretation': interpretation,
                'strategy': strategy,
                'size': cluster_size,
                'percentage': (cluster_size / len(self.df) * 100),
                'avg_recency': avg_recency,
                'avg_frequency': avg_frequency,
                'avg_monetary': avg_monetary,
                'characteristics': {
                    'recency': recency_level,
                    'frequency': frequency_level,
                    'monetary': monetary_level
                }
            }
        
        # Print cluster insights
        print("\n" + "="*80)
        print("CUSTOMER CLUSTER INSIGHTS")
        print("="*80)
        
        for cluster, profile in cluster_profiles.items():
            print(f"\nCluster {cluster}: {profile['interpretation']}")
            print(f"Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
            print(f"Characteristics: {profile['characteristics']['recency']} recency, "
                  f"{profile['characteristics']['frequency']} frequency, "
                  f"{profile['characteristics']['monetary']} monetary value")
            print(f"Averages: {profile['avg_recency']:.0f} days recency, "
                  f"{profile['avg_frequency']:.1f} purchases, "
                  f"${profile['avg_monetary']:.0f} total spent")
            print(f"Strategy: {profile['strategy']}")
            print("-" * 80)
        
        return cluster_profiles

if __name__ == "__main__":
    # Initialize clustering
    clustering = CustomerClustering()
    
    # Find optimal number of clusters
    cluster_analysis = clustering.find_optimal_clusters()
    
    # Perform clustering
    clustered_data, cluster_stats = clustering.perform_clustering()
    
    # Create visualizations
    clustering.visualize_clusters()
    
    # Get cluster interpretations
    cluster_insights = clustering.interpret_clusters()
    
    print("\nClustering analysis completed successfully!")
    print("Files generated:")
    print("- data/processed/clustered_customers.csv")
    print("- data/processed/cluster_statistics.csv")
    print("- visualizations/optimal_clusters_analysis.png")
    print("- visualizations/clustering_analysis.png")