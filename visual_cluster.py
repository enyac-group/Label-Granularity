import pickle
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

results_dir = '/home/rzding/pytorch-cifar/results'
SPLIT = 'test'
DATASET_NAME = 'CIFAR100'
if DATASET_NAME == 'CIFAR10':
	if SPLIT == 'test':
		dataset = 'CIFAR10_TEST'
		file_fine_feat   = '2018-04-28_15-55-33'
		file_coarse_feat = '2018-04-28_15-57-18'
	elif SPLIT == 'train':
		dataset = 'CIFAR10_TRAIN'
		file_fine_feat   = '2018-04-26_23-31-23'
		file_coarse_feat = '2018-04-26_17-55-54'
elif DATASET_NAME == 'CIFAR100':
	if SPLIT == 'test':
		dataset ='CIFAR100_TEST' 
		file_fine_feat   = '2018-04-28_15-59-37'
		file_coarse_feat = '2018-04-28_16-00-17'
	elif SPLIT == 'train':
		dataset = 'CIFAR100_TRAIN'
		file_fine_feat   = '2018-04-26_22-04-00'
		file_coarse_feat = '2018-04-26_13-44-43'
	

np.random.seed(1234)

if DATASET_NAME == 'CIFAR10':
	coarse_classes = ('0','1')

	fine_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	classes_c2f = {'0': ['bird', 'plane', 'car', 'ship', 'truck'],
			'1': ['cat', 'deer', 'dog', 'frog', 'horse']}


elif DATASET_NAME == 'CIFAR100':
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
	#print mean.shape
	distance_to_mean = np.linalg.norm(x-mean, ord=2, axis=1)
	#print distance_to_mean.shape
	sum_of_square = np.square(distance_to_mean).sum()
	#print sum_of_square
	return sum_of_square, np.squeeze(mean)

def compute_class_variance(labels, features, exp_name=''):
	intra_class_dist = 0
	num_sample = 0.
	class_mean = []
	num_class = max(labels)+1
	for label in range(int(num_class)):
		index = (labels == label)
		num_sample += index.sum()
		intra_class_dist_temp, mean_temp = sum_square_distance(features[index,:])
#		print intra_class_dist_temp/(index.sum()-1)
		intra_class_dist += intra_class_dist_temp
		class_mean.append(mean_temp)
	print exp_name, ' intra class variance=', intra_class_dist/(float(num_sample) - num_class)
	print 'Number of samples:', num_sample
	print 'Number of classes:', num_class

	inter_class_dist,_ = sum_square_distance(np.array(class_mean))
	print exp_name, ' inter class variance=', inter_class_dist/(num_class-1.)

	return np.array(class_mean)

print 'Reading file...'
if SPLIT == 'train':
	feat_file_name = 'train_feats.pkl'
elif SPLIT =='test':
	feat_file_name = 'test_feats.pkl' 

with open(os.path.join(results_dir, file_fine_feat, feat_file_name), 'r') as f:
	fine_feat = pickle.load(f)
with open(os.path.join(results_dir, file_fine_feat, 'debug.pkl'), 'r') as f:
	fine_label = np.array(pickle.load(f)[1], dtype=np.float)
#print fine_feat.shape
print dataset, ' GT label fine class:', fine_label
with open(os.path.join(results_dir, file_coarse_feat, feat_file_name), 'r') as f:
	coarse_feat = pickle.load(f)
with open(os.path.join(results_dir, file_coarse_feat, 'debug.pkl'), 'r') as f:
	coarse_label = np.array(pickle.load(f)[1],dtype=np.float)
#print coarse_feat.shape
print dataset, ' GT label coarse class', coarse_label


print 'Generating fine to coarse index map...'
f2c_idx_map=[] # fine to coarse index map
for fine in fine_classes:
	for coarse_label, fine_label_list in classes_c2f.iteritems():
		if fine in fine_label_list:
			f2c_idx_map.append(coarse_classes.index(coarse_label))
			break

coarse_label = np.zeros(fine_label.shape)
for idx, label in enumerate(list(fine_label)):
	coarse_label[idx] = (f2c_idx_map[int(label)])
print dataset, ' GT label mapped coarse class', coarse_label

print set(list(fine_label))
#print set(list(cifar100_100on20_label))
print set(list(coarse_label))

num_fine_class = len(set(list(fine_label)))
num_coarse_class = len(set(list(coarse_label)))

total_size = fine_label.shape[0]
sample_size = min(20000, total_size)
shuffle_idx = np.random.permutation(total_size)[:sample_size]

fine_fc7_rwn1_sample = normalize_r(fine_feat[shuffle_idx,:])
coarse_fc7_rwn1_sample = normalize_r(coarse_feat[shuffle_idx,:])

### compute inter and intra class variance across the entire training dataset
coarse_mean 	= compute_class_variance(coarse_label, normalize_r(coarse_feat), exp_name=dataset+'_coarse')
f2c_mean 	= compute_class_variance(coarse_label, normalize_r(fine_feat), exp_name=dataset+'_f2c')
#print 'f2c_mean distance:', np.linalg.norm(f2c_mean[0,:]-f2c_mean[1,:], ord=2, axis=0)
fine_mean 	= compute_class_variance(fine_label, normalize_r(fine_feat), exp_name=dataset+'_fine')

intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(coarse_feat))
print dataset, ' coarse Overall variance: ', intra_class_dist_temp/(coarse_feat.shape[0])
intra_class_dist_temp, mean_temp = sum_square_distance(normalize_r(fine_feat))
print dataset, 'fine Overall variance: ', intra_class_dist_temp/(fine_feat.shape[0])

	
use_PCA=0
if use_PCA:
	print 'PCA on', dataset+'_fine', '...'
	fine_fc7_rwn1_sample_pca = PCA(n_components=50).fit_transform(fine_fc7_rwn1_sample)
	print 'PCA on', dataset+'coarse', '...'
	coarse_fc7_rwn1_sample_pca  = PCA(n_components=50).fit_transform(coarse_fc7_rwn1_sample)
else:
	fine_fc7_rwn1_sample_pca = np.vstack((fine_fc7_rwn1_sample, fine_mean, f2c_mean))
	coarse_fc7_rwn1_sample_pca  = np.vstack((coarse_fc7_rwn1_sample,coarse_mean))

print 'tSNE on ', dataset+'_fine', '...'
fine_fc7_rwn1_tsne_sample = TSNE().fit_transform(fine_fc7_rwn1_sample_pca)
print 'tSNE on ', dataset+'_coarse', '...'
coarse_fc7_rwn1_tsne_sample = TSNE().fit_transform(coarse_fc7_rwn1_sample_pca)

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
	plot_fig(fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),0], fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),1], mean=[fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class):-(num_coarse_class),0], fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class):-num_coarse_class,1]], label=fine_label[shuffle_idx],filename=dataset+'_fine_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),0], fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),1], label=coarse_label[shuffle_idx], mean=[fine_fc7_rwn1_tsne_sample[-num_coarse_class:,0], fine_fc7_rwn1_tsne_sample[-num_coarse_class:,1]],  filename=dataset+'_f2c_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

	plot_fig(coarse_fc7_rwn1_tsne_sample[:-num_coarse_class,0], coarse_fc7_rwn1_tsne_sample[:-num_coarse_class,1], label=coarse_label[shuffle_idx], mean=[coarse_fc7_rwn1_tsne_sample[-num_coarse_class:,0], coarse_fc7_rwn1_tsne_sample[-num_coarse_class:,1]], filename=dataset+'_coarse_fc7_rwn1_TSNE_sample-{0}.jpg'.format(sample_size))

if plot_density:
	print 'Plotting density...'
	temp_label = fine_label[shuffle_idx]
	for i in range(num_fine_class):
		temp_idx = (temp_label == i)
		plot_density(fine_fc7_rwn1_tsne_sample[temp_idx,0], fine_fc7_rwn1_tsne_sample[temp_idx,1], mean=[fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class)+i,0], fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class)+i,1]], filename=dataset+'_fine_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))
	#print fine_fc7_rwn1_tsne_sample[-2:,0], fine_fc7_rwn1_tsne_sample[-2:,1]

	plot_density(fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),0], fine_fc7_rwn1_tsne_sample[:-(num_fine_class+num_coarse_class),1], mean=[fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class):-num_coarse_class,0], fine_fc7_rwn1_tsne_sample[-(num_fine_class+num_coarse_class):-num_coarse_class,1]], filename=dataset+'_fine_fc7_rwn1_TSNE_sample-{0}_density_allclass.jpg'.format(sample_size))
	
	temp_label = coarse_label[shuffle_idx]
	for i in range(num_coarse_class):
		temp_idx = (temp_label == i)
		plot_density(fine_fc7_rwn1_tsne_sample[temp_idx,0], fine_fc7_rwn1_tsne_sample[temp_idx,1], mean=[fine_fc7_rwn1_tsne_sample[-num_coarse_class+i,0], fine_fc7_rwn1_tsne_sample[-num_coarse_class+i,1]], filename=dataset+'_f2c_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))

		plot_density(coarse_fc7_rwn1_tsne_sample[temp_idx,0], coarse_fc7_rwn1_tsne_sample[temp_idx,1], mean=[coarse_fc7_rwn1_tsne_sample[-num_coarse_class+i,0], coarse_fc7_rwn1_tsne_sample[-num_coarse_class+i,1]], filename=dataset+'_coarse_fc7_rwn1_TSNE_sample-{0}_density_class{1}.jpg'.format(sample_size,i))

