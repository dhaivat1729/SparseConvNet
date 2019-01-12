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

## Facebook's standard network
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
    def __init__(self, root_path, model, criterion, thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

        ### loading the data and all into the os
        self.data_path = root_path
        
        ### getting list of the data
        self.data_list = glob.glob(root_path+'/*.pkl')
        self.data_list.sort()

        ### Number of inputs 
        self.data_length = len(self.data_list)

        ### Getting train and test data in random order
        self.indices = np.random.permutation(self.data_length)

        ### Dividing training and test set in some ratio
        ### For 1000 points, 250 points for training and 750 points for testing, then ratio is 1:3
        self.train_part = 1
        self.test_part = 1

        ### Defining criterion for the neural network
        self.criterion = criterion

        ### getting train and test indices(randomly)
        self.train_indices = self.indices[0:int(self.train_part*self.data_length/(self.train_part + self.test_part))]
        self.test_indices = self.indices[int(self.train_part*self.data_length/(self.train_part + self.test_part)):-1]

        ### Sparsenet model
        self.model = model

        ### Learning algorithm parameters
        ### Learning params
        self.p = {}
        self.p['n_epochs'] = 10
        self.p['initial_lr'] = 1e-1
        self.p['lr_decay'] = 4e-2
        self.p['weight_decay'] = 1e-4
        self.p['momentum'] = 0.9
        # p['check_point'] = False
        self.p['use_cuda'] = torch.cuda.is_available()
        dtype = 'torch.cuda.FloatTensor' if self.p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if self.p['use_cuda'] else 'torch.LongTensor'

        ## IF GPU is available
        if self.p['use_cuda']:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(model)
            self.criterion = self.criterion.cuda()

        ### Training data variables
        self.coords = None
        self.features = None
        self.train_output = None

        ### Optimizer(Defined in training)
        self.optimizer = None

        ### Precision, recall thresholds
        self.thresholds = thresholds

        ### Probability values
        self.ps = None

        ### Precision recall for each point cloud
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.f1_score = None
        self.classes = None
        self.pred_labels = None

        ## confusion matrix
        self.confusion_m = None

        ### To compute average time of forward pass
        self.time_taken = [];

    ### Normalizes the input data and samples points
    def normalize_input(self, df):

        ### Getting coordinates and features
        x = df.iloc[0]['scan_utm']['x'] 
        y = df.iloc[0]['scan_utm']['y']
        z = df.iloc[0]['scan_utm']['z'] 

        ### getting coordinate values between 0 and 150
        x -= min(x)
        y -= min(y)
        z -= min(z)
        x = 150*x/max(x)
        y = 150*y/max(y) 
        z = 150*z/max(z) 

        ### sampling every n th point as point cloud has more than 0.1 million points
        n = 1
        x = x[0::n]
        y = y[0::n]
        z = z[0::n]

        ### Normalizing features?? Not decided yet

        ### final train and test data
        self.coords = torch.randn(len(x), 3)
        self.coords[:,0] = torch.from_numpy(x.copy())
        self.coords[:,1] = torch.from_numpy(y.copy())
        self.coords[:,2] = torch.from_numpy(z.copy())

        ### Getting sampled features
        self.features = df.iloc[0]['scan_utm']['intensity']
        self.features = self.features[0::n]
        self.features = torch.from_numpy(self.features.copy())
        self.features.resize_(len(self.features), 1)
        del x, y, z

        ## Getting the ground truth at every sample
        self.train_output = 1*(df.iloc[0]['is_road_truth'] == True)
        self.train_output = self.train_output[0::n]
        self.train_output = torch.from_numpy(self.train_output.copy())
        #self.train_output = self.train_output.resize_(len(self.train_output), 1)

    def train_model(self):
        
        ### Learning params
        self.p['n_epochs'] = 12
        self.p['initial_lr'] = 1e-1
        self.p['lr_decay'] = 4e-2
        self.p['weight_decay'] = 1e-4
        self.p['momentum'] = 0.9
        # p['check_point'] = False
        self.p['use_cuda'] = torch.cuda.is_available()
        dtype = 'torch.cuda.FloatTensor' if self.p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if self.p['use_cuda'] else 'torch.LongTensor'

        ## IF GPU is available
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

        ### Let's start training
        for epoch in range(self.p['n_epochs']):

            running_loss = 0

            ### Model in a train mode
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
            print("Training on ", len(self.train_indices), " point clouds.")
            for i in self.train_indices:

                ## Keeping count of how many data points are loaded
                steps+=1 
                #if steps%100 == 0: 
                #print("At step: ", steps)

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
                # print("Reached backward pass")

                ## backprop into the loss to compute gradients
                loss.backward()

                ## Updating weights
                self.optimizer.step()

                ## Calculating running loss
                running_loss+= loss.item()
            
            print("Epoch: {}/{}... ".format(epoch+1, self.p['n_epochs']), "Loss: {:.4f}", running_loss/len(self.train_indices))        

            if (epoch+1)%3==0
                self.test_model()

    def test_model(self):

        steps = 0
        
        self.model.eval()

        dtype = 'torch.cuda.FloatTensor' if self.p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if self.p['use_cuda'] else 'torch.LongTensor'
        ## This has total number of true pixels for all the pointclouds being tested
        true_predictions = 0
        ## Total number of points processed across all the pointclouds being tested
        total_points = 0
        ## Let's test for entire test set
        bad_acc_pointclouds = 0   ## number of pointclouds with bad accuracy
    

        print("Testing on: ", len(self.test_indices), "point clouds.")
        lowest_acc = 200
        highest_acc = 10
        for i in self.test_indices:

            ## Keeping count of how many data points are loaded
            steps+=1 
            #print("At step: ", steps)

            ## let's load the data
            df = pd.read_pickle(self.data_list[i])

            ## Prepare data for training
            self.normalize_input(df)
                
            ## Converting input into cuda  tensor if GPU is available
            self.coords = self.coords.type(torch.LongTensor)
            self.features=self.features.type(dtype)
            self.test_output=self.train_output.type(torch.FloatTensor) ### To compute accuracy
            with torch.no_grad():
                ## Forward pass
                t = time.time()
                predictions=self.model((self.coords, self.features))        
                self.time_taken.append(time.time() - t)


            ## Softmax
            self.ps = F.softmax(predictions, dim=1)
            values, index = self.ps.max(dim = 1)
            index = index.type(torch.FloatTensor)
            accuracy = 100*(np.count_nonzero((index - self.test_output).numpy()==0)/len(index))
            #print("Accuracy for point cloud ", i, " is: ", accuracy, "%.")
            if accuracy < 90:
                bad_acc_pointclouds +=1
            if accuracy < lowest_acc:
                lowest_acc = accuracy
            if accuracy > highest_acc:
                highest_acc = accuracy
            true_predictions+= np.count_nonzero((index - self.test_output).numpy()==0)
            total_points+=len(index)

        overall_accuracy = true_predictions/total_points
        print(bad_acc_pointclouds, " point clouds have accuracy less than 90.")
        print("Average time for forward pass is: ", np.mean(np.array(self.time_taken)))
        print("Total accuracy is: ", overall_accuracy*100)
        print("Lowest accuracy is: ", lowest_acc)
        print("Highest accuracy is: ", highest_acc)
        bad_acc_pointclouds = 0
        self.time_taken = []

    def test_single_cloud(self, path):
        ## testing a single point cloud and generating statistics
        self.model.eval()

        ## Datatype based on GPU availability
        dtype = 'torch.cuda.FloatTensor' if self.p['use_cuda'] else 'torch.FloatTensor'
        dtypei = 'torch.cuda.LongTensor' if self.p['use_cuda'] else 'torch.LongTensor'

        ## Loading the point cloud
        df = pd.read_pickle(path)
        
        ## Prepare data for training
        self.normalize_input(df)
            
        ## Converting input into cuda  tensor if GPU is available
        self.coords = self.coords.type(torch.LongTensor)
        self.features=self.features.type(dtype)
        self.test_output=self.train_output.numpy() #.type(torch.FloatTensor) ### To compute accuracy
        with torch.no_grad():
            ## Forward pass
            predictions=self.model((self.coords, self.features))        


        ## Softmax
        self.ps = F.softmax(predictions, dim=1)

        ## let's compute confusion matrix and statistics
        self.confusion()

    ## Compute confusion matrix, precision, recall etc etc for a single pointcloud
    def confusion(self):

        ## precision and recall array for different thresholds
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.accuracy = []
        for thresh in self.thresholds:
            ## create prediction vector
            self.classes = np.where(self.ps[:,1].cpu().numpy()>thresh)
            self.pred_labels = np.zeros(self.ps.shape[0])
            self.pred_labels[self.classes] = 1

            ## Create confusion matrix
            self.confusion_m = np.bincount(self.pred_labels.astype(int)*2+self.test_output.astype(int), minlength=4).reshape((2,2))
            
            ## getting precision,recall from confusion matrix
            tp = self.confusion_m[1,1]   ## True positive
            tn = self.confusion_m[0,0]   ## True negative
            fp = self.confusion_m[1,0]   ## False positive
            fn = self.confusion_m[0,1]   ## False negative
            print("tp, tn, fp, fn: ", tp, tn, fp, fn)
            ## Saving precision and recall
            self.precision.append(tp/(tp+fp))
            self.recall.append(tp/(tp+fn))
            self.f1_score.append(2*self.precision[-1]*self.recall[-1]/(self.precision[-1] + self.recall[-1]))
            self.accuracy.append((tp+tn)/self.ps.shape[0])



        ## Let's compute the confusion matrix.

    ## This saves train and test indices, model, accuracy and other necessary information for future usage
    # def save_variables(self):
    #     ## To be written in future



## Creating a model
# model = torch.load('/home/dhai1729/road_segmentation.model')
model = Model()
print("Model is created!")

## Criterion for the loss
criterion = nn.CrossEntropyLoss()

## Creating object(refine this once it works)
trainobj = train_road_segmentation('/home/dhai1729/maplite_data/data_chunks/', model, criterion, thresholds = np.arange(0.05,1.0,0.05))
print("About to go in training.")
trainobj.train_model()

torch.save(trainobj.model, '/home/dhai1729/road_segmentation_2.model')    



