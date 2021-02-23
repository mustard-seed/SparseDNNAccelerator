import sys, os, time
from glob import glob

import yaml
import scipy.io

NUM_CLASSES = 1000

path_imagenet_labels = 'ILSVRC2012_validation_ground_truth.txt'
path_imagenet_files = 'image_files.txt'
path_synset_words = 'synset_words.txt'
path_meta = 'meta.mat'
path_demo_labels = 'demo_ground_truth.yaml'
path_caffe_words = 'caffe_words.yaml'

meta = scipy.io.loadmat

meta = scipy.io.loadmat(str(path_meta))
original_idx_to_wnid = {}
wnid_to_caffe_idx = {}
wnid_to_text = {}
demo_ground_truth = {}
caffe_to_words = {}
files = []

# Generate a dictionary that maps from the Imagenet IDs to the WNIDs
for i in range(NUM_CLASSES):
	imagenet_id = int(meta["synsets"][i,0][0][0][0])
	wnid = meta["synsets"][i,0][1][0]
	original_idx_to_wnid[imagenet_id] = wnid

# Generate a dictionary that maps from the WNIDs to the PyTorch IDs
with open(path_synset_words, 'r') as f:
	for caffe_idx, line in enumerate(f):
		line = line.strip()
		parts = line.split(' ')
		wnid_to_caffe_idx[parts[0]] = caffe_idx
		text = ' '.join(parts[1:])
		caffe_to_words[caffe_idx] = text

# Generate the ground truth file for the demo
with open(path_imagenet_files, 'r') as f:
	for idx, line in enumerate(f):
		if (idx > 0):
			parts = line.split(' ')
			file = parts[-1].strip()
			files.append(file)

with open(path_imagenet_labels, 'r') as f:
	for idx, line in enumerate(f):
		parts = line.split(' ')
		wnid = original_idx_to_wnid[int(parts[0].strip())]
		demo_ground_truth[files[idx]] = wnid_to_caffe_idx[wnid]

file_demo_labels = open(path_demo_labels, 'w')
yaml.dump(demo_ground_truth, file_demo_labels, default_flow_style=False)
file_caffe_to_text = open(path_caffe_words, 'w')
yaml.dump(caffe_to_words, file_caffe_to_text, default_flow_style=False)