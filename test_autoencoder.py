import numpy as np
import sklearn


train_feat=[]
test_feat = []
train_label=[]
test_label=[]

def test_accuracy_autoencoder(train_feat, test_feat, train_label, test_label):

	Dist = sklearn.metric.pairwise.pairwise_distances(train_feat, test_feat)
	nearest_n = np.argmax(Dist, aixs = 0)
	pred_label = train_label[nearest_n]
	return np.sum(pred_label == test_label)/test_feat.shape[0]
