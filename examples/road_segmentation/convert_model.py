
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


'''

This is a sample code to run maplite pointcloud segmentation. 

Input: pickle file similar to truth.pkl, but contains only 1 pointcloud than all.
Output: Predicted output(1/0 ==> road/non-road) over a sampled point cloud

# You need to load model and pointcloud file at line 67 and 70 respectively.(Provided through a drive link)

Variables you need to care about: (x_org, y_org, z_org, output_classes), all these variables are arrays of same length 
  x_org => x co-ordinates of all the pixels in a sampled pointcloud
  y_org => y co-ordinates of all the pixels in a sampled pointcloud
  z_org => z co-ordinates of all the pixels in a sampled pointcloud
  output_classes => Prediction of road(1) v/s non-road(0) 

  They are all numpy arrays!

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys
import math
import numpy as np
import _pickle as pkl
import numpy as np
import pandas as pd
import glob

model_name = 'small_model_20190612-104041_'
model_data_path = '/home/dhai1729/SparseConvNet/examples/road_segmentation/' + model_name + '.pkl'
model_path = '/home/dhai1729/SparseConvNet/examples/road_segmentation/' + model_name + '.model'
PATH = '/home/dhai1729/SparseConvNet/examples/road_segmentation/' + model_name + '.model.scn'
model_data = pd.read_pickle(model_data_path)
sz = model_data['sz']
spatialSize = model_data['spatialSize']
dimension = 3
reps = model_data['reps']
m = model_data['m']
nPlanes = model_data['nPlanes']
classes_total = 2 # Total number of classes
sampling_factor = model_data['sampling_factor']
use_features = model_data['features'] ## Choices: 'ring', 'z', 'const_vec', 'inten'
num_features = len(use_features)
for n, i in enumerate(use_features):
	if i == 'inten':
		print("Changing inten to intensity!")
		use_features[n] = 'intensity'
norm_fact = model_data['norm_fact']

## Facebook's standard network
class Model(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.sparseModel = scn.Sequential().add(
		   scn.InputLayer(dimension, spatialSize, mode=3)).add(
		   scn.SubmanifoldConvolution(dimension, num_features, m, 3, False)).add(
		   scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2])).add(
		   scn.BatchNormReLU(m)).add(
		   scn.OutputLayer(dimension))
		self.linear = nn.Linear(m, classes_total)
	def forward(self,x):
		x=self.sparseModel(x)
		x=self.linear(x)
		return x

### Loading the model
model = torch.load(model_path)
model_options = {'sz':sz, 'reps':reps, 'num_features':num_features, 'm':m, 'dimension':dimension, 'classes':classes_total, 'nPlanes':nPlanes}
process_options = {'slice':sampling_factor, 'scale':norm_fact, 'feature_list':use_features}
metadata = {'train_path':model_data['train_path'], 'test_path':model_data['test_path'], 'learning_params':model_data['learning_params']}

# PATH = '/home/dhai1729/SparseConvNet/examples/road_segmentation/small_model_20190612-104041_.model.scn'

# or k,v in model.state_dict().items():
	  
#       # k = 'module.' + k

#       # if 'module.' in k:
#       #   name = k[7:]
#       # else:
#       #   name = k
#       new_state_dict[k] = v


### let's save the model in new format!
torch.save({
			'state_dict': model.module.state_dict(),
			'model_options': model_options,
			'process_options': process_options,
			'metadata': metadata
			}, PATH)
