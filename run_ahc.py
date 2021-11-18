import pandas
import matplotlib.pyplot as pyplot
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

dataset = pandas.read_csv("dataset_ahc.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatter_ahc.png")
pyplot.close()

#agglomerative hierarchical clustering uses a bottom-up approach: we consider each individual dot as a cluster (200), then two dots as a cluster by shortest distance (100), and so forth, until we achieve the optimal number of clusters, otherwise we end up with all dots within a single cluster

pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method ='ward'))
pyplot.savefig("dendrogram.png")
pyplot.close()

machine = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
result_ahc = machine.fit_predict(dataset)
pyplot.scatter(dataset['x1'], dataset['x2'], c = result_ahc)
pyplot.savefig("scatter_ahc2.png")
pyplot.close()

#compare it to the kmeans, kmedoids, and gmm