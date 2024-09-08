import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib

data = pd.read_csv(r'your path\Financial_Data_clustering.csv')

text_to_vector = TfidfVectorizer(stop_words = 'english')

text_to_vector.fit(data['S&P Business Description'].astype(str))

vector = text_to_vector.fit_transform(data['S&P Business Description'].astype(str))

optimal_k = []
# finding optimal clusters
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(vector)
    res = kmeans.inertia_
    optimal_k.append(res)

optimal_k


def graph_size(size1, size2, size3, size4):
    matplotlib.rcParams['font.size'] = size1
    # title size
    matplotlib.rcParams['figure.titlesize'] = size2
    # graph size
    matplotlib.rcParams['figure.figsize'] = [size3, size4]

graph_size(10, 15, 13,7)

# Plot
plt.plot(range(2,10), optimal_k, marker= 'o')

# Eblow-method title
plt.title('Elbow-Method k=2-10')
# fontsize --> 'Number of clusters'
plt.xlabel('Number of clusters')
# fontsize --> 'Eblow-value'
plt.ylabel('Eblow-value')
#matplotlib rcParameter initialize
#plt.rcdefaults()
plt.show()


cluster_num = 8
kmeans_optimal = KMeans(n_clusters = cluster_num, random_state = 0)

kmeans_optimal.fit(vector)
# Clustering
y_pred = kmeans_optimal.predict(vector)

data['clustering'] = y_pred

clusters = y_pred

# Extract centroids
order_centroids = kmeans_optimal.cluster_centers_.argsort()[:, ::-1]

# extract key keywords
terms = text_to_vector.get_feature_names()

keywords_df = pd.DataFrame()

for i in range(8):
    cluster_keywords = [terms[ind] for ind in order_centroids[i, :10]]  # select top 10 keywords
    keywords_df[f'Cluster {i}'] = cluster_keywords

# print
print(keywords_df.T)

top_keywords = keywords_df.T

top_keywords.to_csv(r'your_path_to_save_file\important_keywords.csv')