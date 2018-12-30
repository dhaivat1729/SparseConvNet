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
import pandas as pd
import glob

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
        self.data_list = glob.glob(root_path+'/*.pkl')
        self.data_list.sort()

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
        self.train_indices = self.indices[0:int(self.train_part*self.data_length/(self.train_part + self.test_part))]
        self.test_indices = self.indices[int(self.train_part*self.data_length/(self.train_part + self.test_part)):-1]

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
        x = 150*x/max(x)
        y = 150*y/max(y) 
        z = 150*z/max(z) 

        ### Normalizing features?? Not decided yet

        ### final train and test data
        self.coords = torch.randn(len(x), 3)
        self.coords[:,0] = torch.from_numpy(x.copy())
        self.coords[:,1] = torch.from_numpy(y.copy())
        self.coords[:,2] = torch.from_numpy(z.copy())
        self.features = torch.from_numpy(df.iloc[0]['scan_utm']['intensity'].copy())
        self.features.resize_(len(self.features), 1)
        del x, y, z
        self.train_output = torch.from_numpy(1*(df.iloc[0]['is_road_truth'] == True))


    def train_model(self):
        
        ### Learning params
        self.p['n_epochs'] = 100
        self.p['initial_lr'] = 1e-1
        self.p['lr_decay'] = 4e-2
        self.p['weight_decay'] = 1e-4
        self.p['momentum'] = 0.9
        # p['check_point'] = False
        self.p['use_cuda'] = torch.cuda.is_available()
        dtype = 'torch.cuda.FloatTensor' if self.p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if self.p['use_cuda'] else 'torch.LongTensor'

        if self.p['use_cuda']:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(model)
            self.criterion = self.criterion.cuda()

        ### Defining an optimizer for training
        self.optimizer = optim.SGD(self.model.parameters(),
            lr=self.p['initial_lr'],
            momentum = self.p['momentum'],
            weight_decay = self.p['weight_decay'],
            nesterov=True)

        for epoch in range(self.p['n_epochs']):

            running_loss = 0

            ### Model in a train mdoe
            self.model.train()
            # stats = {}

            ### Don't know what this is happening!
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.p['initial_lr'] * math.exp((1 - epoch) * self.p['lr_decay'])
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()

            ### let's start the training
            ### iterating through the dataset and training
            steps = 0
            for i in self.train_indices:

                ## Keeping count of how many data points are loaded
                steps+=1 
                print("At step: ", steps)

                ## let's load the data
                df = pd.read_pickle(self.data_list[i])

                ## Prepare data for training
                self.normalize_input(df)
                # break
                self.optimizer.zero_grad()

                ## Converting input into cuda  tensor if GPU is available
                self.coords = self.coords.type(torch.LongTensor)
                self.features=self.features.type(dtype)
                self.train_output=self.train_output.type(dtypei)
                
                ## Forward pass
                predictions=self.model((self.coords, self.features))
                # print(predictions.max(), predictions.min())

                ## Computing loss
                loss = self.criterion.forward(predictions,self.train_output)

                ## backprop into the loss to compute gradients
                loss.backward()

                ## Updating weights
                self.optimizer.step()

                ## Calculating running loss
                running_loss+= loss.item()
            
                print("Epoch: {}/{}... ".format(epoch+1, self.p['n_epochs']), "Loss: {:.4f}".format(loss.item()))        

    def test_model(self):

        steps = 0
        
        self.model.eval()

        ## Let's test for entire test set
        for i in self.test_indices:

                ## Keeping count of how many data points are loaded
                steps+=1 
                print("At step: ", steps)

                ## let's load the data
                df = pd.read_pickle(self.data_list[i])

                ## Prepare data for training
                self.normalize_input(df)
                
                ## Converting input into cuda  tensor if GPU is available
                self.coords = self.coords.type(torch.LongTensor)
                self.features=self.features.type(dtype)
                self.train_output=self.train_output.type(dtypei)
                with torch.no_grad():
                    ## Forward pass
                    predictions=self.model((self.coords, self.features))        


                ## Softmax
                ps = F.softmax(predictions, dim=1)
                values, index = ps.max(dim = 1)


## Creating a model
model=Model()
print("Model is created!")

## Criterion for the loss
criterion = nn.CrossEntropyLoss()

## Creating object(refine this once it works)
trainobj = train_road_segmentation('/home/dhai1729/maplite_data/data_chunks/', model, criterion)
print("About to go in training.")
trainobj.train_model()
torch.save(trainobj.model, '/home/dhai1729/road_segmentation.model')



