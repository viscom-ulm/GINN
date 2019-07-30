'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GIEval.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import time
import argparse
import importlib
import os
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/utils'))

from PyUtils import visualize_progress, save_model
from GIDataSet import GIDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_loss(predictVals, ptsVals, dataset):
    diffVals = tf.subtract(predictVals, ptsVals)
    diffVals = tf.square(diffVals)
    valLoss = tf.reduce_mean(diffVals)
    valLoss = tf.sqrt(valLoss)
    return valLoss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of the GI networks')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--outFolder', default='outFolder', help='Output folder (default: outFolder)')
    parser.add_argument('--dataset', default=0, type=int, help='Data set used (0 - AO, 1 - GI, 2 - SS) (default: 0)')
    parser.add_argument('--model', default='MCGINetworkRTEval', help='model (default: MCGINetworkRTEval)')
    parser.add_argument('--grow', default=8, type=int, help='Grow rate (default: 8)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    parser.add_argument('--nExec', default=1, type=int, help='Number of executions (default: 1)')
    args = parser.parse_args()

    if not os.path.exists(args.outFolder): os.mkdir(args.outFolder)

    print("DataSet: "+str(args.dataset))
    print("Trained model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    testDataSet = GIDataSet(2, args.dataset, 1, False, True)
    numTestModels = testDataSet.get_num_models()
    print("Test models: " + str(numTestModels))

    #Create variable and place holders
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    inFeatures = tf.placeholder(tf.float32, [None, testDataSet.get_feature_channels()])
    inPts2 = tf.placeholder(tf.float32, [None, 3])
    inBatchIds2 = tf.placeholder(tf.int32, [None, 1])
    if args.dataset == 2:
        inFeatures2 = tf.placeholder(tf.float32, [None, testDataSet.get_feature_channels()])
    else:
        inFeatures2 = tf.placeholder(tf.float32, [None, 3])
    inGI = tf.placeholder(tf.float32, [None, testDataSet.get_label_channels()])
    isTraining = tf.placeholder(tf.bool, shape=())

    #Create the network
    brnClipping = { 'rmax': 10.0,
                    'rmin': 1.0/1.0,
                    'dmax': 5.0}
    predVals = model.create_network(
        inPts, inBatchIds, inFeatures, 
        inPts2, inBatchIds2, inFeatures2,
        inGI, 1, args.grow, isTraining, 
        1.0, 1.0, brnClipping,
        1.0, 1.0, False, False, args.dataset)
          
    #Create loss
    loss = create_loss(predVals, inGI, args.dataset)

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Init variables
    sess.run(init, {isTraining: True})
    sess.run(initLocal, {isTraining: True})

    #Restore the model
    saver.restore(sess, args.inTrainedModel)
    
    #Test data
    accumTestLoss = 0.0
    for i in range(args.nExec):
        testDataSet.start_iteration()
        for it in range(numTestModels):
                
            points, batchIds, features, gi = testDataSet.get_next_batch()

            lossRes, predValsRes = sess.run([loss, predVals], {
                inPts: points, inBatchIds: batchIds, inFeatures: features, 
                inPts2: points, inBatchIds2: batchIds, inFeatures2: features, inGI: gi, 
                isTraining: False})
            
            accumTestLoss += lossRes

            save_model(args.outFolder+"/"+str(it)+"_pred", points, np.clip(predValsRes, 0.0, 1.0))
            save_model(args.outFolder+"/"+str(it)+"_gt", points, np.clip(gi, 0.0, 1.0))
            
            if it%100 == 0:
                visualize_progress(it, numTestModels*args.nExec)

    accumTestLoss = accumTestLoss/float(numTestModels*args.nExec)


    print("Loss: %.6f" % (accumTestLoss))
