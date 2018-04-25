import argparse
import pickle
import numpy as np
import os


parser = argparse.ArgumentParser(description="Parse pickle file and analyze clustering")
parser.add_argument('dir', help='directory of input')
args = parser.parse_args()
input_dir = args.dir


# with open('label_clustered2.pkl', 'r') as f:
# 	label_clustered = pickle.load(f)

# with open('label_superclass2.pkl', 'r') as f:
# 	label_superclass= pickle.load(f)

with open('results/label_true2.pkl') as f:
	label_true = pickle.load(f)

with open(os.path.join('results', dir, 'label_f.pkl'), 'r') as f:
	label_clustered = pickle.load(f)

#with open('label_shuffle_clustered2.pkl', 'r') as f:
#	label_shuffle_clustered = pickle.load(f)

#print label_clustered
#print label_superclass
#print label_true
# output:
# [ 8.  1.  3. ...,  0.  2.  3.]
# [array([    0,     1,     2, ..., 49997, 49998, 49999]), array([1, 0, 0, ..., 0, 0, 0])]
# [array([    0,     1,     2, ..., 49997, 49998, 49999]), array([6, 9, 9, ..., 9, 1, 1])]

## class mapping
## classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
## superclass0:'bird','plane','car','ship','truck'
## superclass1:'cat','deer','dog','frog','horse'

file_idx = label_superclass[0]
original_label = label_true[1]
superclass_mask = label_superclass[1]
clustered_label = np.array(label_clustered)
# clustered_label_wisely = np.array(label_wisely_clustered)
# clustered_label_shuffle = np.array(label_shuffle_clustered)


#superclass_0_idx = np.where(superclass_mask == 0)[0]

print 'Results from ',input_dir
for class_idx in range(int(max(original_label))+1):
	hist_2D = []
	origin_class_k_idx = np.where(original_label == class_idx)[0]
	hist,bin_edges = np.histogram(clustered_label[origin_class_k_idx], bins=np.arange(max(original_label)+1))
	hist_2D.append(hist)
	print 'original class ', class_idx, ': ', hist, ' row sum: ', sum(hist), 'max idx: ', np.argmax(hist)
	print 'Sum of columns: ', np.sum(np.array(hist_2D), axis=0)
	#print bin_edges

#print 'labels from train 10 correct 5:5 partition'
#for class_idx in range(int(max(original_label))+1):
#	origin_class_k_idx = np.where(original_label == class_idx)[0]
#	hist,bin_edges = np.histogram(clustered_label_wisely[origin_class_k_idx], bins=np.arange(11))
#	print 'original class ', class_idx, ': ', hist, ' sum: ', sum(hist), 'max idx: ', np.argmax(hist)
#	#print bin_edges
#
#
#print 'labels from train 10 shuffle 5:5 partition'
#for class_idx in range(int(max(original_label))+1):
#	origin_class_k_idx = np.where(original_label == class_idx)[0]
#	hist,bin_edges = np.histogram(clustered_label_shuffle[origin_class_k_idx], bins=np.arange(11))
#	print 'original class ', class_idx, ': ', hist, ' sum: ', sum(hist), 'max idx: ', np.argmax(hist)
#	#print bin_edges
