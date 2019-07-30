'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GIDataSet.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import time
import numpy as np
from sklearn.preprocessing import normalize
from os import listdir
from os.path import isfile, join


class GIDataSet():
    """GI dataset.
    """
    
    def __init__(self, dataset, giData, batchSize, augment=False, allChannels=False, seed=None):
        """Constructor.

        Args:
            dataset (int): Boolean that indicates if this is the train or test dataset.
                - 0: training
                - 1: evaluation
                - 2: testing 
            giData (int): Gi data that should be used to learn. 
                - 0: Ambient occlusion
                - 1: Diffuse interactions
                - 2: Subsurface scattering 
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            batchSize (int): Size of the batch used.
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Check if the dataset is valid.
        if not(((dataset>=0)and(dataset<3))or((giData>=0)and(giData<3))):
            raise RuntimeError('Invalid dataset')

        # Store the model list.
        self.tetDataset_ = dataset
        dataFolders = ["ao_data", "gi_data", "sss_data"]
        datasets = ["training", "evaluation", "test"]
        numPts = [10240, 20000, 15000]
        self.fileList_ = [join(dataFolders[giData]+"/"+datasets[dataset]+"/", f) 
            for f in listdir(dataFolders[giData]+"/"+datasets[dataset]) 
            if isfile(join(dataFolders[giData]+"/"+datasets[dataset]+"/", f)) 
            and f.endswith(".npy")]
        self.numPts_ = numPts[giData]

        self.data_ = []
        for fileIter, curFile in enumerate(self.fileList_):
            fileData = np.load(curFile)
            pts = fileData[:self.numPts_,0:3]
            coordMax = np.amax(pts, axis=0)
            coordMin = np.amin(pts, axis=0)
            center = (coordMax+coordMin)*0.5
            sizeAABB = coordMax - coordMin
            maxSize = np.amax(sizeAABB)
            if giData == 2:
                pts = (pts - center)
            else:
                pts = (pts - center)/maxSize
            fileData[:self.numPts_,3:6] = normalize(fileData[:self.numPts_,3:6], axis=1)
            if giData == 0:
                gi = fileData[:self.numPts_,6]
            elif giData == 1:
                features = fileData[:self.numPts_,3:12]
                gi = fileData[:self.numPts_,12:]
            elif giData == 2:
                features = fileData[:self.numPts_,3:16]
                gi = fileData[:self.numPts_,16:]
            self.data_.append((pts, features, gi, maxSize))
                    
        # Compute feature or label channels
        self.numFeatureChannels_ = 3
        if allChannels:
            self.numLabelChannels_ = 3
            if giData == 1:
                self.numFeatureChannels_ = 9
            elif giData == 2:
                self.numFeatureChannels_ = 13
        else:
            self.numLabelChannels_ = 1
            if giData == 1:
                self.numFeatureChannels_ = 5
            elif giData == 2:
                self.numFeatureChannels_ = 7

        # Store the dataset used
        self.giData_ = giData

        # Store if we need to augment the dataset.
        self.augment_ = augment

        # Initialize the random seed.
        if not(seed is None):
            self.randomState_ = np.random.RandomState(seed)
        else:
            self.randomState_ = np.random.RandomState(int(time.time()))

        self.randomSelection_ = []
        self.iterator_ = 0

        self.batchSize_ = batchSize

        self.allChannels_ = allChannels


    def get_num_models(self):
        """Method to consult the number of models in the dataset.

        Returns:
            numModels (int): Number of models in the dataset.
        """
        return len(self.fileList_)

    def get_feature_channels(self):
        """Method to get the number of feature channels.

        Returns:
            numChannels (int): Number of feature channels.
        """
        return self.numFeatureChannels_


    def get_label_channels(self):
        """Method to get the number of label channels.

        Returns:
            numChannels (int): Number of label channels.
        """
        return self.numLabelChannels_

    def start_iteration(self):
        """Method to start an iteration over the models.
        """
        self.randomSelection_ = self.randomState_.permutation(len(self.fileList_))
        self.iterator_ = 0

    def get_next_batch(self):

        batchPts = []
        batchFeatures = []
        batchBatchIds = []
        batchGI = []
        for i in range(self.batchSize_):
            if self.iterator_ < len(self.randomSelection_):
                curIndex = self.randomSelection_[self.iterator_]
                self.iterator_ += 1

                pts = self.data_[curIndex][0]
                features = self.data_[curIndex][1]
                gi = self.data_[curIndex][2]
                maxSize = self.data_[curIndex][3]

                if self.augment_:
                    angles = 3.141516*self.randomState_.randn(3)
                    Ry = np.array([[np.cos(angles[1]), 0.0, np.sin(angles[1])],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angles[1]), 0.0, np.cos(angles[1])]])
                    pts = np.dot(pts, Ry)
                    features[:, 0:3] = np.dot(features[:, 0:3], Ry)

                if not self.allChannels_:
                    if self.giData_ == 1:
                        rndChannel = int(math.floor(self.randomState_.uniform(0.0, 3.0)))
                        features = features[:, [0,1,2,3+rndChannel,6+rndChannel]]
                        gi = gi[:, rndChannel]
                    elif self.giData_ == 2:
                        rndChannel = int(math.floor(self.randomState_.uniform(0.0, 3.0)))
                        features = features[:, [0,1,2,3+rndChannel,6+rndChannel,9+rndChannel,12]]
                        gi = gi[:, rndChannel]
 
                if self.tetDataset_ == 0:
                    ptsNoise = self.randomState_.normal(0.0, 0.01, pts.shape)
                    pts = pts + ptsNoise
                    noise = self.randomState_.normal(0.0, 0.05, gi.shape)
                    gi = gi + noise

                batchPts.append(pts)
                batchFeatures.append(features)
                batchBatchIds.append(np.full([len(pts),1], i, dtype = int))
                batchGI.append(gi)

        batchPts = np.concatenate(tuple(batchPts),axis=0)
        batchFeatures = np.concatenate(tuple(batchFeatures),axis=0)
        batchBatchIds = np.concatenate(tuple(batchBatchIds),axis=0)
        batchGI = np.concatenate(tuple(batchGI),axis=0)
        if not self.allChannels_:
            batchGI = batchGI.reshape((-1, 1))
        else:
            batchGI = batchGI.reshape((-1, 3))

        return batchPts, batchBatchIds, batchFeatures, batchGI