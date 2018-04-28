import pickle
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

results_dir = '/home/rzding/pytorch-cifar/results'
file_cifar100_100_feature = '2018-04-26_23-31-23'
file_cifar100_20_feature = '2018-04-26_17-55-54'

np.random.seed(1234)

coarse_classes = ('0','1')

fine_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

classes_c2f = {'0': ['bird', 'plane', 'car', 'ship', 'truck'],
		'1': ['cat', 'deer', 'dog', 'frog', 'horse']}


def normalize_r(x):
     return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def sum_square_distance(x):
	mean = np.mean(x, axis=0, keepdims=True)
	distance_to_mean = np.linalg.norm(x-mean, ord=2, axis=1)
	return np.sum(np.square(distance_to_mean)), np.squeeze(mean)

def compute_class_variance(labels, features, exp_name=''):
	intra_class_dist = 0
	num_sample = len(list(labels))
	class_mean = []
	num_class = max(labels)+1
	for label in set(list(labels)):
		index = (labels == label)
		intra_class_dist_temp, mean_temp = sum_square_distance(features[index,:])
#		print intra_class_dist_temp/(index.sum()-1)
		intra_class_dist += intra_class_dist_temp
		class_mean.append(mean_temp)
	print exp_name, ' intra class variance=', intra_class_dist/(float(num_sample) - num_class)

	inter_class_dist,_ = sum_square_distance(np.array(class_mean))
	print exp_name, ' inter class variance=', inter_class_dist/(num_class-1.)

	return np.array(class_mean)

print 'Reading file...'
with open(os.path.join(results_dir, file_cifar100_100_feature, 'train_feats.pkl'), 'r') as f:
	cifar100_100_feature = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_100_feature, 'debug.pkl'), 'r') as f:
	cifar100_100_label = np.array(pickle.load(f)[1], dtype=np.float)
#print cifar100_100_feature.shape
print 'cifar100 GT label 10 fine class:', cifar100_100_label
with open(os.path.join(results_dir, file_cifar100_20_feature, 'train_feats.pkl'), 'r') as f:
	cifar100_20_feature = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_20_feature, 'debug.pkl'), 'r') as f:
	cifar100_20_label = np.array(pickle.load(f)[1],dtype=np.float)
#print cifar100_20_feature.shape
print 'cifar100 GT label 2 coarse class', cifar100_20_label

print 'Generating fine to coarse index map...'
f2c_idx_map=[] # fine to coarse index map
for fine in fine_classes:
	for coarse_label, fine_label_list in classes_c2f.iteritems():
		if fine in fine_label_list:
			f2c_idx_map.append(coarse_classes.index(coarse_label))
			break

cifar100_100on20_label = np.zeros(cifar100_100_label.shape)
for idx, label in enumerate(list(cifar100_100_label)):
	cifar100_100on20_label[idx] = (f2c_idx_map[int(label)])

print set(list(cifar100_100_label))
print set(list(cifar100_100on20_label))
print set(list(cifar100_20_label))


sample_size = 20000
total_size = cifar100_100_label.shape[0]
shuffle_idx = np.random.permutation(total_size)[:sample_size]

cifar100_100_fc7_rwn1 = normalize_r(cifar100_100_feature[shuffle_idx,:])
cifar100_20_fc7_rwn1 = normalize_r(cifar100_20_feature[shuffle_idx,:])

### compute inter and intra class variance across the entire training dataset
coarse_mean 	= compute_class_variance(cifar100_20_label, normalize_r(cifar100_20_feature), exp_name='cifar10_2')
f2c_mean 	= compute_class_variance(cifar100_20_label, normalize_r(cifar100_100_feature), exp_name='cifar10_10on2')
#print 'f2c_mean distance:', np.linalg.norm(f2c_mean[0,:]-f2c_mean[1,:], ord=2, axis=0)
fine_mean 	= compute_class_variance(cifar100_100_label, normalize_r(cifar100_100_feature), exp_name='cifar10_10')

intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(cifar100_20_feature))
print 'cifar10_2 Overall variance: ', intra_class_dist_temp/(cifar100_20_feature.shape[0])
intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(cifar100_100_feature))
print 'cifar10_10 Overall variance: ', intra_class_dist_temp/(cifar100_100_feature.shape[0])

	
use_PCA=0
if use_PCA:
	print 'PCA on CIFAR10_10...'
	cifar100_100_fc7_rwn1_pca = PCA(n_components=50).fit_transform(cifar100_100_fc7_rwn1)
	print 'PCA on CIFAR10_2...'
	cifar100_20_fc7_rwn1_pca  = PCA(n_components=50).fit_transform(cifar100_20_fc7_rwn1)
else:
	cifar100_100_fc7_rwn1_pca = np.vstack((cifar100_100_fc7_rwn1, fine_mean, f2c_mean))
	cifar100_20_fc7_rwn1_pca  = np.vstack((cifar100_20_fc7_rwn1,coarse_mean))

print 'tSNE on CIFAR10_10...'
cifar100_100_fc7_rwn1_embedded = TSNE().fit_transform(cifar100_100_fc7_rwn1_pca)
print 'tSNE on CIFAR10_2...'
cifar100_20_fc7_rwn1_embedded = TSNE().fit_transform(cifar100_20_fc7_rwn1_pca)

plot_data=1
plot_density=1

def plot_fig(x, y, label, mean=[],  filename='noname.jpg'):
	plt.scatter(x, y, c=label, cmap=plt.cm.Spectral)
	plt.scatter(mean[0], mean[1], s=100, c='r', marker='^')
	plt.savefig(filename)
	
def plot_density(x, y, mean=[], filename='noname.jpg'):
	plt.figure()
	plt.hist2d(x, y, (50,50),cmap=plt.cm.jet)
	plt.colorbar()
	plt.scatter(mean[0], mean[1], s=100, c='r', marker='^')
	plt.savefig(filename)

if plot_data:
	print 'Plotting projected data...'
	plot_fig(cifar100_100_fc7_rwn1_embedded[:-12,0], cifar100_100_fc7_rwn1_embedded[:-12,1], mean=[cifar100_100_fc7_rwn1_embedded[-12:-2,0], cifar100_100_fc7_rwn1_embedded[-12:-2,1]], label=cifar100_100_label[shuffle_idx],filename='cifar10_10_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(cifar100_100_fc7_rwn1_embedded[:-12,0], cifar100_100_fc7_rwn1_embedded[:-12,1], label=cifar100_100on20_label[shuffle_idx], mean=[cifar100_100_fc7_rwn1_embedded[-2:,0], cifar100_100_fc7_rwn1_embedded[-2:,1]],  filename='cifar10_10on2_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(cifar100_20_fc7_rwn1_embedded[:-2,0], cifar100_20_fc7_rwn1_embedded[:-2,1], label=cifar100_20_label[shuffle_idx], mean=[cifar100_20_fc7_rwn1_embedded[-2:,0], cifar100_20_fc7_rwn1_embedded[-2:,1]], filename='cifar10_2_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

if plot_density:
	print 'Plotting density...'
	#temp_label = cifar100_100_label[shuffle_idx]
	#for i in range(10):
	#	temp_idx = (temp_label == i)
	plot_density(cifar100_100_fc7_rwn1_embedded[:-12,0], cifar100_100_fc7_rwn1_embedded[:-12,1], mean=[cifar100_100_fc7_rwn1_embedded[-12:-2,0], cifar100_100_fc7_rwn1_embedded[-12:-2,1]], filename='cifar10_10_fc7_rwn1_TSNE_sample-{0}_density.jpg'.format(sample_size))
	#print cifar100_100_fc7_rwn1_embedded[-2:,0], cifar100_100_fc7_rwn1_embedded[-2:,1]
	plot_density(cifar100_100_fc7_rwn1_embedded[:-12,0], cifar100_100_fc7_rwn1_embedded[:-12,1], mean=[cifar100_100_fc7_rwn1_embedded[-2:,0], cifar100_100_fc7_rwn1_embedded[-2:,1]], filename='cifar10_10on2_fc7_rwn1_TSNE_sample-{0}_density.jpg'.format(sample_size))
	plot_density(cifar100_20_fc7_rwn1_embedded[:-2,0], cifar100_20_fc7_rwn1_embedded[:-2,1], mean=[cifar100_20_fc7_rwn1_embedded[-2:,0], cifar100_20_fc7_rwn1_embedded[-2:,1]], filename='cifar10_2_fc7_rwn1_TSNE_sample-{0}_density.jpg'.format(sample_size))

