'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCGINetworkRTEval.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import tensorflow as tf
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import conv_1x1

def compute_initial_pts(points, batchIds, features, batchSize):

    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.01], "MCGIInit_PH1", batchSize, relativeRadius=False)
    return mPointHierarchy.sampledIndexs_[0], mPointHierarchy.points_[1]


def create_network(points, batchIds, features,  
    points2, batchIds2, features2, outputs,
    batchSize, k, isTraining, 
    bnMomentum, brnMomentum, brnClipping,
    keepProbConv, keepProbFull, 
    useConvDropOut = False, useDropOutFull = True,
    dataset = 0):

    def BN_NL_DP_Conv(layerName, inFeatures):
        inFeatures = tf.layers.batch_normalization(inputs = inFeatures, momentum=bnMomentum, 
            trainable = True, training = isTraining, name = layerName+"_BN", renorm=True,
            renorm_clipping=brnClipping, renorm_momentum=brnMomentum)
        inFeatures = tf.nn.leaky_relu(inFeatures)
        if useConvDropOut:
            inFeatures = tf.nn.dropout(inFeatures, keepProbConv)
        return inFeatures

    def BN_NL_DP_F(layerName, inFeatures):
        inFeatures = tf.layers.batch_normalization(inputs = inFeatures, momentum=bnMomentum, 
            trainable = True, training = isTraining, name = layerName+"_BN", renorm=True,
            renorm_clipping=brnClipping, renorm_momentum=brnMomentum)
        inFeatures = tf.nn.leaky_relu(inFeatures)
        if useDropOutFull:
            inFeatures = tf.nn.dropout(inFeatures, keepProbFull)
        return inFeatures

    ############################################  Compute point hierarchy
    mPointHierarchy1 = PointHierarchy(points, features, batchIds, [0.05, 0.1, 0.2], "MCGI_PH1", batchSize, relativeRadius=False)
    mPointHierarchy2 = PointHierarchy(points2, features2, batchIds2, [], "MCGI_PH2", batchSize, relativeRadius=False)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25, relativeRadius=False)

    resultColor = []
    with tf.variable_scope("feature_channel_scope", reuse=tf.AUTO_REUSE):
        for i in range(3):

            if dataset == 1:
                numInFeatures = 5
                inputFeatures = features[:,0:3]
                inputFeatures = tf.concat([inputFeatures, features[:,3+i:3+i+1]], axis=1)
                inputFeatures = tf.concat([inputFeatures, features[:,6+i:6+i+1]], axis=1)
                inputFeatures2 = features2[:,0:3]
            elif dataset == 2:
                numInFeatures = 7
                inputFeatures = features[:,0:3]
                inputFeatures = tf.concat([inputFeatures, features[:,3+i:3+i+1]], axis=1)
                inputFeatures = tf.concat([inputFeatures, features[:,6+i:6+i+1]], axis=1)
                inputFeatures = tf.concat([inputFeatures, features[:,9+i:9+i+1]], axis=1)
                inputFeatures = tf.concat([inputFeatures, features[:,12:13]], axis=1)
                inputFeatures2 = features2[:,0:3]
                inputFeatures2 = tf.concat([inputFeatures2, features2[:,3+i:3+i+1]], axis=1)
                inputFeatures2 = tf.concat([inputFeatures2, features2[:,6+i:6+i+1]], axis=1)
                inputFeatures2 = tf.concat([inputFeatures2, features2[:,9+i:9+i+1]], axis=1)
                inputFeatures2 = tf.concat([inputFeatures2, features2[:,12:13]], axis=1)
            else:
                numInFeatures = 3
                inputFeatures = features 
                inputFeatures2 = features2    

            ############################################ Encoder

            # First convolution
            convFeatures1 = mConvBuilder.create_convolution(
                convName="Conv_1", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=0, 
                inFeatures=inputFeatures,
                inNumFeatures=numInFeatures, 
                outNumFeatures=k,
                convRadius=0.025,
                multiFeatureConv=True)

            # First pooling
            bnConvFeatures1 = BN_NL_DP_Conv("Reduce_Pool_1_In_BN", convFeatures1)
            bnConvFeatures1 = conv_1x1("Reduce_Pool_1", bnConvFeatures1, k, k*2)
            bnConvFeatures1 = BN_NL_DP_Conv("Reduce_Pool_1_Out_BN", bnConvFeatures1)
            poolFeatures1 = mConvBuilder.create_convolution(
                convName="Pool_1", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=0, 
                outPointLevel=1, 
                inFeatures=bnConvFeatures1,
                inNumFeatures=k*2, 
                convRadius=0.05,
                KDEWindow= 0.2)

            # Second convolution
            bnPoolFeatures1 = BN_NL_DP_Conv("Reduce_Conv_2_In_BN", poolFeatures1)
            bnPoolFeatures1 = conv_1x1("Reduce_Conv_2", bnPoolFeatures1, k*2, k*2)
            bnPoolFeatures1 = BN_NL_DP_Conv("Reduce_Conv_2_Out_BN", bnPoolFeatures1)
            convFeatures2 = mConvBuilder.create_convolution(
                convName="Conv_2", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=1, 
                inFeatures=bnPoolFeatures1,
                inNumFeatures=k*2, 
                convRadius=0.1)
            convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)

            # Second pooling
            bnConvFeatures2 = BN_NL_DP_Conv("Reduce_Pool_2_In_BN", convFeatures2)
            bnConvFeatures2 = conv_1x1("Reduce_Pool_2", bnConvFeatures2, k*4, k*4)
            bnConvFeatures2 = BN_NL_DP_Conv("Reduce_Pool_2_Out_BN", bnConvFeatures2)
            poolFeatures2 = mConvBuilder.create_convolution(
                convName="Pool_2", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=1, 
                outPointLevel=2, 
                inFeatures=bnConvFeatures2,
                inNumFeatures=k*4, 
                convRadius=0.2,
                KDEWindow= 0.2)

            # Third convolution
            bnPoolFeatures2 = BN_NL_DP_Conv("Reduce_Conv_3_In_BN", poolFeatures2)
            bnPoolFeatures2 = conv_1x1("Reduce_Conv_3", bnPoolFeatures2, k*4, k*4)
            bnPoolFeatures2 = BN_NL_DP_Conv("Reduce_Conv_3_Out_BN", bnPoolFeatures2)
            convFeatures3 = mConvBuilder.create_convolution(
                convName="Conv_3", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=2, 
                inFeatures=bnPoolFeatures2,
                inNumFeatures=k*4, 
                convRadius=0.2)
            convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

            # Third pooling
            bnConvFeatures3 = BN_NL_DP_Conv("Reduce_Pool_3_In_BN", convFeatures3)
            bnConvFeatures3 = conv_1x1("Reduce_Pool_3", bnConvFeatures3, k*8, k*8)
            bnConvFeatures3 = BN_NL_DP_Conv("Reduce_Pool_3_Out_BN", bnConvFeatures3)
            poolFeatures3 = mConvBuilder.create_convolution(
                convName="Pool_3", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=2, 
                outPointLevel=3, 
                inFeatures=bnConvFeatures3,
                inNumFeatures=k*8, 
                convRadius=0.4,
                KDEWindow= 0.2)

            # Fourth convolution
            bnPoolFeatures3 = BN_NL_DP_Conv("Reduce_Conv_4_In_BN", poolFeatures3)
            bnPoolFeatures3 = conv_1x1("Reduce_Conv_4", bnPoolFeatures3, k*8, k*8)
            bnPoolFeatures3 = BN_NL_DP_Conv("Reduce_Conv_4_Out_BN", bnPoolFeatures3)
            convFeatures4 = mConvBuilder.create_convolution(
                convName="Conv_4", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=3, 
                inFeatures=bnPoolFeatures3,
                inNumFeatures=k*8, 
                convRadius=1.0,
                KDEWindow= 0.2)

            ############################################ Decoder

            # Third upsampling
            bnConvFeatures4 = BN_NL_DP_Conv("Up_3_BN", convFeatures4)
            upFeatures3 = mConvBuilder.create_convolution(
                convName="Up_3", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=3, 
                outPointLevel=2, 
                inFeatures=bnConvFeatures4,
                inNumFeatures=k*8, 
                convRadius=0.4,
                KDEWindow= 0.2)
            upFeatures3 = tf.concat([convFeatures3, upFeatures3], 1)
            upFeatures3 = BN_NL_DP_Conv("Up_3_Reduce_BN", upFeatures3)
            upFeatures3 = conv_1x1("Up_3_Reduce", upFeatures3, k*16, k*4)

            # Second upsampling
            bnUpFeatures3 = BN_NL_DP_Conv("Up_2_BN", upFeatures3)
            upFeatures2 = mConvBuilder.create_convolution(
                convName="Up_2", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=2, 
                outPointLevel=1, 
                inFeatures=bnUpFeatures3,
                inNumFeatures=k*4, 
                convRadius=0.2,
                KDEWindow= 0.2)
            upFeatures2 = tf.concat([convFeatures2, upFeatures2], 1)
            upFeatures2 = BN_NL_DP_Conv("Up_2_Reduce_BN", upFeatures2)
            upFeatures2 = conv_1x1("Up_2_Reduce", upFeatures2, k*8, k*2)
            
            # First upsampling
            bnUpFeatures2 = BN_NL_DP_Conv("Up_1_2_BN", upFeatures2)
            upFeatures1 = mConvBuilder.create_convolution(
                convName="Up_1_2", 
                inPointHierarchy=mPointHierarchy1,
                inPointLevel=1, 
                outPointLevel=0, 
                inFeatures=bnUpFeatures2,
                inNumFeatures=k*2, 
                convRadius=0.1,
                KDEWindow= 0.2)
            upFeatures1 = tf.concat([convFeatures1, upFeatures1], 1)
            upFeatures1 = BN_NL_DP_Conv("Up_1_Reduce_BN", upFeatures1)
            upFeatures1 = conv_1x1("Up_1_Reduce", upFeatures1, k*3, k)


            bnUpFeatures1 = BN_NL_DP_Conv("Up_1_BN", upFeatures1)
            finalFeatures = mConvBuilder.create_convolution(
                convName="Final_Conv", 
                inPointHierarchy=mPointHierarchy1,
                outPointHierarchy=mPointHierarchy2,
                inPointLevel=0, 
                outPointLevel=0, 
                inFeatures=bnUpFeatures1,
                inNumFeatures=k, 
                convRadius=0.05,
                KDEWindow= 0.2)

            finalFeatures = BN_NL_DP_F("Final_MLP1_BN", finalFeatures)
            finalFeatures = tf.concat([finalFeatures, inputFeatures2], 1)
            finalFeatures = conv_1x1("Final_MLP1", finalFeatures, k + numInFeatures, k)
            finalFeatures = BN_NL_DP_F("Final_MLP2_BN", finalFeatures)
            predVals = conv_1x1("Final_MLP2", finalFeatures, k, 1)
            resultColor.append(predVals)

    return tf.concat(resultColor, axis=1)
