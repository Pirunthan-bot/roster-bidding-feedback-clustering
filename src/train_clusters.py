import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

file_path = 'data/data_bidding_remarks.csv'
output_path = 'output/clustered_bidding_remarks.csv'

# Load the CSV file
data = pd.read_csv(file_path)

# Preprocess the remarks
remarks = data['Bidding Remarks']
remarks_cleaned = remarks.str.lower()  # Convert text to lowercase

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(remarks_cleaned)

# Apply KMeans clustering
num_clusters = 5 
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

# Assign each remark to a cluster
data['Cluster'] = kmeans.labels_

# Save the output to a new CSV file
data.to_csv(output_path, index=False)

# Display a sample of the clustered remarks
print(data[['Bidding Remarks', 'Cluster']].head(10))
