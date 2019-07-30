'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GITrainRT.py

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

from PyUtils import visualize_progress
from GIDataSet import GIDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_loss(predictVals, ptsVals, weigthDecay):
    diffVals = tf.subtract(predictVals, ptsVals)
    diffVals = tf.square(diffVals)
    valLoss = tf.reduce_mean(diffVals)
    valLoss = tf.sqrt(valLoss)
    regularizer = tf.contrib.layers.l2_regularizer(scale=weigthDecay)
    regVariables = tf.get_collection('weight_decay_loss')
    regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    return valLoss, regTerm


def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp

def create_bn_momentum(InitBNtDecay, bnDecayRate, bnDecayFactor, maxBNDecay, global_step):
    bnDecayExp = tf.train.exponential_decay(InitBNtDecay, global_step, bnDecayRate, bnDecayFactor, staircase=True)
    bnDecayExp = tf.maximum(bnDecayExp, maxBNDecay)
    return 1.0-bnDecayExp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train GINN')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCGINetworkRT', help='model (default: MCGINetworkRT)')
    parser.add_argument('--grow', default=8, type=int, help='Grow rate (default: 8)')
    parser.add_argument('--dataset', default=0, type=int, help='Data set used (0 - AO, 1 - GI, 2 - SS) (default: 0)')
    parser.add_argument('--batchSize', default=16, type=int, help='Batch size  (default: 16)')
    parser.add_argument('--maxEpoch', default=201, type=int, help='Max Epoch  (default: 201)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDeacyFactor', default=0.7, type=float, help='Learning deacy factor (default: 0.7)')
    parser.add_argument('--learningDecayRate', default=10, type=int, help='Learning decay rate  (default: 10 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.000001, type=float, help='Maximum Learning rate (default: 0.000001)')
    parser.add_argument('--initBNDecay', default=0.5, type=float, help='Init batch normalization momentum  (default: 0.5)')
    parser.add_argument('--BNDecayFactor', default=0.5, type=float, help='Batch normalization deacy factor (default: 0.5)')
    parser.add_argument('--BNDecayRate', default=10, type=int, help='Batch normalization decay rate  (default: 10 Epochs)')
    parser.add_argument('--maxBNDecay', default=0.01, type=float, help='Maximum batch normalization decay (default: 0.01)')
    parser.add_argument('--useDropOut', action='store_true', help='Use drop out  (default: False)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probabillity drop out  (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use drop out in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.9, type=float, help='Keep neuron probabillity drop out in convolution layers (default: 0.9)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='Weight decay (default: 0.0)')
    parser.add_argument('--ptDropOut', default=1.0, type=float, help='Point drop out (default: 1.0)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.75, type=float, help='GPU memory used (default: 0.75)')
    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp %s.py %s' % (args.model, args.logFolder))
    os.system('cp GI.py %s' % (args.logFolder))
    logFile = args.logFolder+"/log.txt"

    #Write execution info.
    with open(logFile, "a") as myFile:
        myFile.write("Model: "+args.model+"\n")
        myFile.write("Grow: "+str(args.grow)+"\n")
        myFile.write("DataSet: "+str(args.dataset)+"\n")
        myFile.write("BatchSize: "+str(args.batchSize)+"\n")
        myFile.write("MaxEpoch: "+str(args.maxEpoch)+"\n")
        myFile.write("InitLearningRate: "+str(args.initLearningRate)+"\n")
        myFile.write("LearningDeacyFactor: "+str(args.learningDeacyFactor)+"\n")
        myFile.write("LearningDecayRate: "+str(args.learningDecayRate)+"\n")
        myFile.write("MaxLearningRate: "+str(args.maxLearningRate)+"\n")
        myFile.write("UseDropOut: "+str(args.useDropOut)+"\n")
        myFile.write("DropOutKeepProb: "+str(args.dropOutKeepProb)+"\n")
        myFile.write("UseDropOutConv: "+str(args.useDropOutConv)+"\n")
        myFile.write("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv)+"\n")
        myFile.write("WeightDecay: "+str(args.weightDecay)+"\n")
        myFile.write("ptDropOut: "+str(args.ptDropOut)+"\n")
        myFile.write("Augment: "+str(args.augment)+"\n")

    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("DataSet: "+str(args.dataset))
    print("BatchSize: "+str(args.batchSize))
    print("MaxEpoch: "+str(args.maxEpoch))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("LearningDeacyFactor: "+str(args.learningDeacyFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("MaxLearningRate: "+str(args.maxLearningRate))
    print("UseDropOut: "+str(args.useDropOut))
    print("DropOutKeepProb: "+str(args.dropOutKeepProb))
    print("UseDropOutConv: "+str(args.useDropOutConv))
    print("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv))
    print("WeightDecay: "+str(args.weightDecay))
    print("ptDropOut: "+str(args.ptDropOut))
    print("Augment: "+str(args.augment))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    trainDataSet = GIDataSet(0, args.dataset, args.batchSize, args.augment)
    evalDataSet = GIDataSet(1, args.dataset, 1, False)
    numTrainModels = trainDataSet.get_num_models()
    numBatchesXEpoch = numTrainModels//args.batchSize
    numTestModels = evalDataSet.get_num_models()
    print("Train models: " + str(numTrainModels))
    print("Test models: " + str(numTestModels))

    #Create variable and place holders
    global_step = tf.Variable(0, name='global_step', trainable=False)
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    inFeatures = tf.placeholder(tf.float32, [None, trainDataSet.get_feature_channels()])
    inPts2 = tf.placeholder(tf.float32, [None, 3])
    inBatchIds2 = tf.placeholder(tf.int32, [None, 1])
    if args.dataset == 2:
        inFeatures2 = tf.placeholder(tf.float32, [None, trainDataSet.get_feature_channels()])
    else:
        inFeatures2 = tf.placeholder(tf.float32, [None, 3])
    inGI = tf.placeholder(tf.float32, [None, trainDataSet.get_label_channels()])
    isTraining = tf.placeholder(tf.bool, shape=())
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)

    #Create the batch norm momentum.
    bnMomentum = create_bn_momentum(args.initBNDecay, args.BNDecayRate*numBatchesXEpoch, args.BNDecayFactor, args.maxBNDecay, global_step)
    brnMomentum = create_bn_momentum(args.initBNDecay, args.BNDecayRate*numBatchesXEpoch, args.BNDecayFactor, args.maxBNDecay, global_step)
    initBRNDecayEpoch = float(args.BNDecayRate*numBatchesXEpoch)
    endBRNDecayEpoch = float((args.maxEpoch-args.BNDecayRate)*numBatchesXEpoch)
    lerpBRN = 1.0 - tf.clip_by_value((tf.to_float(global_step)-initBRNDecayEpoch)/(endBRNDecayEpoch-initBRNDecayEpoch), 0.0, 1.0)
    rmax = lerpBRN + (1.0-lerpBRN)*5.0
    dmax = (1.0-lerpBRN)*10.0
    brnClipping = { 'rmax': rmax,
                    'rmin': 1.0/rmax,
                    'dmax': dmax}

    #Create the network
    numInFeatures2 = 3
    if args.dataset == 2:
        numInFeatures2 = trainDataSet.get_feature_channels()
    ptIndexs, ptIndexPts = model.compute_initial_pts(inPts, inBatchIds, inFeatures, args.batchSize)
    predVals = model.create_network(
        inPts, inBatchIds, inFeatures, 
        trainDataSet.get_feature_channels(), numInFeatures2, 
        inPts2, inBatchIds2, inFeatures2,
        inGI, trainDataSet.get_label_channels(),
        args.batchSize, args.grow, isTraining, 
        bnMomentum, brnMomentum, brnClipping,
        keepProbConv, keepProbFull, 
        args.useDropOutConv, 
        args.useDropOut, args.dataset)
          
    #Create loss
    valLoss, regularizationLoss = create_loss(predVals, inGI, args.weightDecay)
    loss = valLoss + regularizationLoss

    #Create training
    trainning, learningRateExp = create_trainning(loss, 
        args.initLearningRate, args.maxLearningRate, args.learningDeacyFactor, 
        args.learningDecayRate*numBatchesXEpoch, global_step)
    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)

    #Create sumaries
    lossSummaryPH = tf.placeholder(tf.float32)
    lossSummary = tf.summary.scalar('loss', loss)
    valLossSummary = tf.summary.scalar('loss_Val', valLoss)
    regularizationLossSummary = tf.summary.scalar('loss_Regularization', regularizationLoss)
    trainingSummary = tf.summary.merge([lossSummary, valLossSummary, regularizationLossSummary, learningRateSumm])
    metricsTestSummary = tf.summary.scalar('Tes_loss_AO', lossSummaryPH)

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver(max_to_keep=100)
    
    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Create the summary writer
    summary_writer = tf.summary.FileWriter(args.logFolder, sess.graph)
    summary_writer.add_graph(sess.graph)
    
    #Init variables
    sess.run(init, {isTraining: True})
    sess.run(initLocal, {isTraining: True})
    step = 0
    epochStep = 0
    minValLoss = 1.0
    np.random.seed(int(time.time()))

    #Train
    for epoch in range(args.maxEpoch):

        startEpochTime = current_milli_time()

        epochStep = 0
        lossInfoCounter = 0
        lossAccumValue = 0.0
        accumLoadTime = 0.0
        accumTrainTime = 0.0

        #Iterate over all the train files
        trainDataSet.start_iteration()
        for curBatchIter in range(numBatchesXEpoch):
    
            startLoadTime = current_milli_time()
            points, batchIds, features, gi = trainDataSet.get_next_batch() 

            ptIndexsPtsRes, ptIndexsRes = sess.run([ptIndexPts, ptIndexs], {inPts: points, inBatchIds: batchIds, inFeatures: features})

            ptIndexsRes = ptIndexsRes.flatten()
            ptIndexsSet = set(ptIndexsRes.tolist())
            ptIndexsRes2 = np.array([i for i in range(len(points)) if i not in ptIndexsSet])
            ptIndexsRes2 = np.random.choice(ptIndexsRes2, min(2000, len(ptIndexsRes2)), replace=False)

            auxPts1 = points[ptIndexsRes]
            auxBatchIds1 = batchIds[ptIndexsRes]
            auxfeatures1 = features[ptIndexsRes]

            auxPts2 = points[ptIndexsRes2]
            auxBatchIds2 = batchIds[ptIndexsRes2]
            if args.dataset == 2:
                auxfeatures2 = features[ptIndexsRes2]
            else:
                auxfeatures2 = features[ptIndexsRes2, 0:3]
            auxGI2 = gi[ptIndexsRes2]
            endLoadTime = current_milli_time()
            accumLoadTime += (endLoadTime-startLoadTime)/1000.0
            
            startTrainTime = current_milli_time()
            _, lossRes, valLossRes, regularizationLossRes, trainingSummRes = \
                sess.run([trainning, loss, valLoss, regularizationLoss, trainingSummary], 
                {inPts: auxPts1, inBatchIds: auxBatchIds1, inFeatures: auxfeatures1, 
                 inPts2: auxPts2, inBatchIds2: auxBatchIds2, inFeatures2: auxfeatures2, inGI: auxGI2,
                 isTraining: True, keepProbConv: args.dropOutKeepProbConv, keepProbFull: args.dropOutKeepProb})
            endTrainTime = current_milli_time()
            accumTrainTime += (endTrainTime-startTrainTime)/1000.0
            
            summary_writer.add_summary(trainingSummRes, step)

            lossAccumValue += valLossRes
            lossInfoCounter += 1

            if lossInfoCounter == 10:
                endTrainTime = current_milli_time()

                visualize_progress(epochStep, numBatchesXEpoch, "Loss: %.6f | Time: %.4f (%.4f)" % (
                    lossAccumValue/10.0, accumTrainTime/10.0, accumLoadTime/10.0))

                with open(logFile, "a") as myfile:
                    myfile.write("Step: %6d (%4d) | Loss: %.6f\n" % (step, epochStep, lossAccumValue/10.0))

                lossInfoCounter = 0
                lossAccumValue = 0.0
                accumTrainTime = 0.0
                accumLoadTime = 0.0

            step += 1
            epochStep += 1

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))

        if epoch%10==0:
            saver.save(sess, args.logFolder+"/check_model.ckpt", global_step=epoch)

        #Test data
        accumTestLoss = 0.0
        it = 0
        evalDataSet.start_iteration()
        for curTestFile in range(numTestModels):
                
            points, batchIds, features, gi = evalDataSet.get_next_batch()

            lossRes = sess.run(valLoss, {
                inPts: points, inBatchIds: batchIds, inFeatures: features, 
                inPts2: points, inBatchIds2: batchIds, inFeatures2: features, inGI: gi, 
                isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            
            accumTestLoss += lossRes
            
            if it%100 == 0:
                visualize_progress(it, numTestModels)

            it += 1

        accumTestLoss = accumTestLoss/float(numTestModels)

        metricsTestSummRes = sess.run(metricsTestSummary, {lossSummaryPH: accumTestLoss, isTraining: False})
        summary_writer.add_summary(metricsTestSummRes, step)

        print("Loss: %.6f [ %.4f ]" % (accumTestLoss, minValLoss))
        with open(logFile, "a") as myfile:
            myfile.write("Loss: %.6f [  %.4f ] \n" % (accumTestLoss, minValLoss))
    
        if accumTestLoss < minValLoss:
            minValLoss = accumTestLoss
            saver.save(sess, args.logFolder+"/model.ckpt", global_step=epoch)
