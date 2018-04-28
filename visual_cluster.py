import pickle
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

results_dir = '/home/rzding/pytorch-cifar/results'
file_cifar100_100_feature = '2018-04-26_22-04-00'
file_cifar100_20_feature = '2018-04-26_13-44-43'
classmap = [0,0,0,1,1,1,1,1,0,0]

np.random.seed(1234)

coarse_classes = ('aquatic_mammals', 'fish', 'flowers', 'food_containers',
	'fruit_and_vegetables', 'household_electrical_devices',
	'household_furniture', 'insects', 'large_carnivores',
	'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
	'large_omnivores_and_herbivores', 'medium_mammals',
	'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
	'trees', 'vehicles_1', 'vehicles_2')

fine_classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
		'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
		'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
		'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
		'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
		'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
		'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
		'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
		'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
		'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
		'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
		'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
		'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
		'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
		'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
		'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
		'willow_tree', 'wolf', 'woman', 'worm')

classes_c2f = {'aquatic_mammals': ['beaver','dolphin','otter','seal','whale'],
		'fish': ['aquarium_fish','flatfish','ray','shark','trout'],
		'flowers': ['orchid','poppy','rose','sunflower','tulip'],
		'food_containers': ['bottle','bowl','can','cup','plate'],
		'fruit_and_vegetables': ['apple','mushroom','orange','pear','sweet_pepper'],
		'household_electrical_devices': ['clock','keyboard','lamp','telephone','television'],
		'household_furniture': ['bed','chair','couch','table','wardrobe'],
		'insects': ['bee','beetle','butterfly','caterpillar','cockroach'],
		'large_carnivores': ['bear','leopard','lion','tiger','wolf'],
		'large_man-made_outdoor_things': ['bridge','castle','house','road','skyscraper'],
		'large_natural_outdoor_scenes': ['cloud','forest','mountain','plain','sea'],
		'large_omnivores_and_herbivores': ['camel','cattle','chimpanzee','elephant','kangaroo'],
		'medium_mammals': ['fox','porcupine','possum','raccoon','skunk'],
		'non-insect_invertebrates': ['crab','lobster','snail','spider','worm'],
		'people': ['baby','boy','girl','man','woman'],
		'reptiles': ['crocodile','dinosaur','lizard','snake','turtle'],
		'small_mammals': ['hamster','mouse','rabbit','shrew','squirrel'],
		'trees': ['maple_tree','oak_tree','palm_tree','pine_tree','willow_tree'],
		'vehicles_1': ['bicycle','bus','motorcycle','pickup_truck','train'],
		'vehicles_2': ['lawn_mower','rocket','streetcar','tank','tractor']}


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
		print label, index.sum() 
		intra_class_dist_temp, mean_temp = sum_square_distance(features[index,:])
#		print intra_class_dist_temp/(index.sum()-1)
		intra_class_dist += intra_class_dist_temp
		class_mean.append(mean_temp)
	print exp_name, ' intra class variance=', intra_class_dist/(float(num_sample) - num_class)

	#print np.array(class_mean).shape
	inter_class_dist,_ = sum_square_distance(np.array(class_mean))
	print exp_name, ' inter class variance=', inter_class_dist/(num_class-1.)

print 'Reading file...'
with open(os.path.join(results_dir, file_cifar100_100_feature, 'train_feats.pkl'), 'r') as f:
	cifar100_100_feature = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_100_feature, 'debug.pkl'), 'r') as f:
	cifar100_100_label = np.array(pickle.load(f)[1])
#print cifar100_100_feature.shape
print 'cifar100 GT label 100 fine class:', cifar100_100_label
with open(os.path.join(results_dir, file_cifar100_20_feature, 'train_feats.pkl'), 'r') as f:
	cifar100_20_feature = pickle.load(f)
with open(os.path.join(results_dir, file_cifar100_20_feature, 'debug.pkl'), 'r') as f:
	cifar100_20_label = np.array(pickle.load(f)[1])
#print cifar100_20_feature.shape
print 'cifar100 GT label 20 coarse class', cifar100_20_label

print 'Generating fine to coarse index map...'
f2c_idx_map=[] # fine to coarse index map
for fine in fine_classes:
	for coarse_label, fine_label_list in classes_c2f.iteritems():
		if fine in fine_label_list:
			f2c_idx_map.append(coarse_classes.index(coarse_label))
			break
print 'f2c_idx_map size', np.array(f2c_idx_map).shape

cifar100_100on20_label = np.zeros(cifar100_100_label.shape)
for idx, label in enumerate(list(cifar100_100_label)):
	cifar100_100on20_label[idx] = int(f2c_idx_map[label])
print 'cifar100_100on20_label first 100:', cifar100_100on20_label[:100]

print set(list(cifar100_100_label))
print set(list(cifar100_100on20_label))
print set(list(cifar100_20_label))


sample_size = 10000
total_size = cifar100_100_label.shape[0]
shuffle_idx = np.random.permutation(total_size)[:sample_size]

cifar100_100_fc7_rwn1 = normalize_r(cifar100_100_feature[shuffle_idx,:])
cifar100_20_fc7_rwn1 = normalize_r(cifar100_20_feature[shuffle_idx,:])

### compute inter and intra class variance across the entire training dataset
compute_class_variance(cifar100_20_label, normalize_r(cifar100_20_feature), exp_name='cifar100_20')
compute_class_variance(cifar100_100on20_label, normalize_r(cifar100_100_feature), exp_name='cifar100_100on20')
compute_class_variance(cifar100_100_label, normalize_r(cifar100_100_feature), exp_name='cifar100_100')

intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(cifar100_20_feature))
print 'cifar100_20 Overall variance: ', intra_class_dist_temp/(cifar100_20_feature.shape[0])
intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(cifar100_100_feature))
print 'cifar100_100 Overall variance: ', intra_class_dist_temp/(cifar100_100_feature.shape[0])
	
use_PCA=0
if use_PCA:
	print 'PCA on CIFAR100_100...'
	cifar100_100_fc7_rwn1_pca = PCA(n_components=50).fit_transform(cifar100_100_fc7_rwn1)
	print 'PCA on CIFAR100_20...'
	cifar100_20_fc7_rwn1_pca = PCA(n_components=50).fit_transform(cifar100_20_fc7_rwn1)
else:
	cifar100_100_fc7_rwn1_pca = cifar100_100_fc7_rwn1
	cifar100_20_fc7_rwn1_pca  = cifar100_20_fc7_rwn1

print 'tSNE on CIFAR100_100...'
cifar100_100_fc7_rwn1_embedded = TSNE().fit_transform(cifar100_100_fc7_rwn1_pca)
print 'tSNE on CIFAR100_20...'
cifar100_20_fc7_rwn1_embedded = TSNE().fit_transform(cifar100_20_fc7_rwn1_pca)

print 'Plotting...'
plt.scatter(cifar100_100_fc7_rwn1_embedded[:,0], cifar100_100_fc7_rwn1_embedded[:,1], c=cifar100_100_label[shuffle_idx], cmap=plt.cm.Spectral)
plt.savefig('cifar100_100_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))
plt.scatter(cifar100_100_fc7_rwn1_embedded[:,0], cifar100_100_fc7_rwn1_embedded[:,1], c=cifar100_100on20_label[shuffle_idx], cmap=plt.cm.Spectral)
plt.savefig('cifar100_100on20_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))
plt.scatter(cifar100_20_fc7_rwn1_embedded[:,0], cifar100_20_fc7_rwn1_embedded[:,1], c=cifar100_20_label[shuffle_idx], cmap=plt.cm.Spectral)
plt.savefig('cifar100_20_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))
