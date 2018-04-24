import sklearn.cluster as cls

def clustering(train_data, num_clusters)
	cluster_algo = cls.SpectralClustering(n_clusters=num_clusters)	
	cluster_algo.fit(train_data)
	return cluster_algo.labels_

