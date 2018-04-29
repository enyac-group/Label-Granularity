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

#coarse_classes = ('0','1')
#
#fine_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#classes_c2f = {'0': ['bird', 'plane', 'car', 'ship', 'truck'],
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
	fine_feat = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_100_feature, 'debug.pkl'), 'r') as f:
	fine_label = np.array(pickle.load(f)[1], dtype=np.float)
#print fine_feat.shape
print 'cifar100 GT label 10 fine class:', fine_label
with open(os.path.join(results_dir, file_cifar100_20_feature, 'train_feats.pkl'), 'r') as f:
	coarse_feat = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_20_feature, 'debug.pkl'), 'r') as f:
	coarse_label = np.array(pickle.load(f)[1],dtype=np.float)
#print coarse_feat.shape
print 'cifar100 GT label 2 coarse class', coarse_label

print 'Generating fine to coarse index map...'
f2c_idx_map=[] # fine to coarse index map
for fine in fine_classes:
	for coarse_label, fine_label_list in classes_c2f.iteritems():
		if fine in fine_label_list:
			f2c_idx_map.append(coarse_classes.index(coarse_label))
			break

#cifar100_100on20_label = np.zeros(fine_label.shape)
#for idx, label in enumerate(list(fine_label)):
#	cifar100_100on20_label[idx] = (f2c_idx_map[int(label)])
#
print set(list(fine_label))
#print set(list(cifar100_100on20_label))
print set(list(coarse_label))


sample_size = 20000
total_size = fine_label.shape[0]
shuffle_idx = np.random.permutation(total_size)[:sample_size]

cifar100_100_fc7_rwn1 = normalize_r(fine_feat[shuffle_idx,:])
cifar100_20_fc7_rwn1 = normalize_r(coarse_feat[shuffle_idx,:])

### compute inter and intra class variance across the entire training dataset
coarse_mean 	= compute_class_variance(coarse_label, normalize_r(coarse_feat), exp_name='cifar10_2')
f2c_mean 	= compute_class_variance(coarse_label, normalize_r(fine_feat), exp_name='cifar10_10on2')
#print 'f2c_mean distance:', np.linalg.norm(f2c_mean[0,:]-f2c_mean[1,:], ord=2, axis=0)
fine_mean 	= compute_class_variance(fine_label, normalize_r(fine_feat), exp_name='cifar10_10')

intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(coarse_feat))
print 'cifar10_2 Overall variance: ', intra_class_dist_temp/(coarse_feat.shape[0])
intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(fine_feat))
print 'cifar10_10 Overall variance: ', intra_class_dist_temp/(fine_feat.shape[0])

	
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
fine_fc7_rwn1_tsne = TSNE().fit_transform(cifar100_100_fc7_rwn1_pca)
print 'tSNE on CIFAR10_2...'
coarse_fc7_rwn1_tsne = TSNE().fit_transform(cifar100_20_fc7_rwn1_pca)

plot_data=1
plot_density=1

def plot_fig(x, y, label, mean=[],  filename='noname.jpg'):
	plt.figure()
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
	plot_fig(fine_fc7_rwn1_tsne[:-12,0], fine_fc7_rwn1_tsne[:-12,1], mean=[fine_fc7_rwn1_tsne[-12:-2,0], fine_fc7_rwn1_tsne[-12:-2,1]], label=fine_label[shuffle_idx],filename='cifar10_10_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(fine_fc7_rwn1_tsne[:-12,0], fine_fc7_rwn1_tsne[:-12,1], label=coarse_label[shuffle_idx], mean=[fine_fc7_rwn1_tsne[-2:,0], fine_fc7_rwn1_tsne[-2:,1]],  filename='cifar10_10on2_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(coarse_fc7_rwn1_tsne[:-2,0], coarse_fc7_rwn1_tsne[:-2,1], label=coarse_label[shuffle_idx], mean=[coarse_fc7_rwn1_tsne[-2:,0], coarse_fc7_rwn1_tsne[-2:,1]], filename='cifar10_2_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

if plot_density:
	print 'Plotting density...'
	temp_label = fine_label[shuffle_idx]
	for i in range(10):
		temp_idx = (temp_label == i)
		plot_density(fine_fc7_rwn1_tsne[temp_idx,0], fine_fc7_rwn1_tsne[temp_idx,1], mean=[fine_fc7_rwn1_tsne[-12+i,0], fine_fc7_rwn1_tsne[-12+i,1]], filename='cifar10_10_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))
	#print fine_fc7_rwn1_tsne[-2:,0], fine_fc7_rwn1_tsne[-2:,1]

	plot_density(fine_fc7_rwn1_tsne[:-12,0], fine_fc7_rwn1_tsne[:-12,1], mean=[fine_fc7_rwn1_tsne[-12:-2,0], fine_fc7_rwn1_tsne[-12:-2,1]], filename='cifar10_10_fc7_rwn1_TSNE_sample-{0}_density_allclass.jpg'.format(sample_size))
	
	temp_label = coarse_label[shuffle_idx]
	for i in range(2):
		temp_idx = (temp_label == i)
		plot_density(fine_fc7_rwn1_tsne[temp_idx,0], fine_fc7_rwn1_tsne[temp_idx,1], mean=[fine_fc7_rwn1_tsne[-2+i,0], fine_fc7_rwn1_tsne[-2+i,1]], filename='cifar10_10on2_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))

		plot_density(coarse_fc7_rwn1_tsne[temp_idx,0], coarse_fc7_rwn1_tsne[temp_idx,1], mean=[coarse_fc7_rwn1_tsne[-2+i,0], coarse_fc7_rwn1_tsne[-2+i,1]], filename='cifar10_2_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))

