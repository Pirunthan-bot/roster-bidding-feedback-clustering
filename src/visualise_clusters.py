import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Load the clustered data
data = pd.read_csv('output/clustered_bidding_remarks.csv')

# Recreate the TF-IDF matrix (same as in clustering script)
remarks_cleaned = data['Bidding Remarks'].str.lower()
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(remarks_cleaned)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2, random_state=42)
reduced_features_pca = pca.fit_transform(tfidf_matrix.toarray())

# Reduce dimensions using t-SNE (alternative visualization)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_features_tsne = tsne.fit_transform(tfidf_matrix.toarray())

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. PCA Scatter Plot
scatter1 = axes[0, 0].scatter(reduced_features_pca[:, 0], 
                               reduced_features_pca[:, 1], 
                               c=data['Cluster'], 
                               cmap='viridis', 
                               alpha=0.6,
                               edgecolors='black',
                               linewidth=0.5)
axes[0, 0].set_title('PCA: Bidding Remarks Clusters', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Principal Component 1')
axes[0, 0].set_ylabel('Principal Component 2')
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

# 2. t-SNE Scatter Plot
scatter2 = axes[0, 1].scatter(reduced_features_tsne[:, 0], 
                               reduced_features_tsne[:, 1], 
                               c=data['Cluster'], 
                               cmap='viridis', 
                               alpha=0.6,
                               edgecolors='black',
                               linewidth=0.5)
axes[0, 1].set_title('t-SNE: Bidding Remarks Clusters', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('t-SNE Component 1')
axes[0, 1].set_ylabel('t-SNE Component 2')
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# 3. Cluster Distribution
cluster_counts = data['Cluster'].value_counts().sort_index()
axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='black')
axes[1, 0].set_title('Distribution of Remarks Across Clusters', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of Remarks')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Top keywords per cluster
axes[1, 1].axis('off')
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get top 5 keywords for each cluster
text_output = "Top Keywords per Cluster:\n\n"
for cluster_num in sorted(data['Cluster'].unique()):
    cluster_remarks = data[data['Cluster'] == cluster_num]['Bidding Remarks']
    cluster_text = ' '.join(cluster_remarks.str.lower())
    
    cluster_tfidf = tfidf_vectorizer.transform([cluster_text])
    top_indices = cluster_tfidf.toarray()[0].argsort()[-5:][::-1]
    top_keywords = [feature_names[i] for i in top_indices]
    
    text_output += f"Cluster {cluster_num}: {', '.join(top_keywords)}\n"
    text_output += f"  ({len(cluster_remarks)} remarks)\n\n"

axes[1, 1].text(0.1, 0.9, text_output, 
                fontsize=10, 
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('output/cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'cluster_visualization.png'")
print("\n" + text_output)