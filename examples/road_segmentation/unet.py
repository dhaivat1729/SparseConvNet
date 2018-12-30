# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
# import data
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
import os
import pandas as pd

# data.init(-1,24,24*8,16)
sz = 24*8
spatialSize = torch.LongTensor([sz]*3)
dimension = 3
reps = 1 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level
classes_total = 2 # Total number of classes

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, spatialSize, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
           scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2])).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, classes_total)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x

## Class for data loading, training and testing

class train_road_segmentation():
    def __init__(self, root_path, model, criterion):

        ### loading the data and all into the os
        self.data_path = root_path
        
        ### getting list of the data
        self.data_list = os.listdir(self.data_path).sort()

        ### Number of inputs 
        self.data_length = len(self.data_list)

        ### Randomly sampling training and test data
        self.indices = np.random.permutation(self.data_length)

        ### Dividing training and test set in some ratio
        ### For 1000 points, 250 points for training and 750 points for testing, then ratio is 1:3
        self.train_part = 1
        self.test_part = 3

        ### Defining criterion for the neural network
        self.criterion = criterion

        ### getting train and test indices
        self.train_indices = self.indices[0:self.train_part*self.data_length/(self.train_part + self.test_part)]
        self.test_indices = self.indices[self.train_part*self.data_length/(self.train_part + self.test_part):-1]

        ### Learning algorithm parameters
        self.p = {}

        ### Training data variables
        self.coords = None
        self.features = None
        self.train_output = None

        ### Optimizer(Defined in training)
        self.optimizer = None

        ### Sparsenet model
        self.model = model

    ### Normalizes the input data between -1 to 1
    def normalize_input(self, df):

        ### Getting coordinates and features
        x = df.iloc[0]['scan_utm']['x'] 
        y = df.iloc[0]['scan_utm']['y']
        z = df.iloc[0]['scan_utm']['z'] 

        ### getting coordinate values between -1 and 1
        x -= min(x)
        y -= min(y)
        z -= min(z)
        x = 2*x/max(x) -1
        y = 2*y/max(y) -1
        z = 2*z/max(z) -1

        ### Normalizing features?? Not decided yet

        ### final train and test data
        self.coords = torch.randn(len(x), 3)
        self.coords[:,0] = torch.from_numpy(x.copy())
        self.coords[:,1] = torch.from_numpy(y.copy())
        self.coords[:,2] = torch.from_numpy(z.copy())
        self.features = torch.from_numpy(df.iloc[0]['scan_utm']['intensity'].copy())
        del x, y, z
        self.train_output = 1*(df.iloc[0]['is_road_truth'] == True)


    def train_model(self):
        
        ### Learning params
        p['n_epochs'] = 100
        p['initial_lr'] = 1e-1
        p['lr_decay'] = 4e-2
        p['weight_decay'] = 1e-4
        p['momentum'] = 0.9
        # p['check_point'] = False
        p['use_cuda'] = torch.cuda.is_available()
        dtype = 'torch.cuda.FloatTensor' if p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if p['use_cuda'] else 'torch.LongTensor'

        if p['use_cuda']:
            self.model.cuda()
            self.criterion.cuda()

        ### Defining an optimizer for training
        self.optimizer = optim.SGD(self.model.parameters(),
            lr=p['initial_lr'],
            momentum = p['momentum'],
            weight_decay = p['weight_decay'],
            nesterov=True)

        for epoch in range(p['n_epochs']):

            running_loss = 0

            ### Model in a train mdoe
            self.model.train()
            # stats = {}

            ### Don't know what this is happening!
            for param_group in optimizer.param_groups:
                param_group['lr'] = p['initial_lr'] * math.exp((1 - epoch) * p['lr_decay'])
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()

            ### let's start the training
            ### iterating through the dataset and training
            for i in train_indices:

                ## let's load the data
                df = pd.read_pickle(self.data_path + data_list[i])

                ## Prepare data for training
                self.normalize_input(df)
                optimizer.zero_grad()

                ## Converting input into cuda  tensor if GPU is available
                self.coords=self.coords.type(dtype)
                self.features=self.features.type(dtypei)

                ## Forward pass
                predictions=self.model((self.coords, self.features))

                ## Computing loss
                loss = self.criterion.forward(predictions,train_output)

                ## backprop into the loss to compute gradients
                loss.backward()

                ## Updating weights
                optimizer.step()

                ## Calculating running loss
                running_loss+= loss.item()
            
                print("Epoch: {}/{}... ".format(e+1, p['n_epochs']), "Loss: {:.4f}".format(running_loss/30))          



## Creating a model
model=Model()

## Criterion for the loss
criterion = nn.CrossEntropyLoss()

## Creating object(refine this once it works)
trainobj = train_road_segmentation('/home/dhai1729/maplite_data/data_chunks/', model, criterion)
trainobj.train_model()




