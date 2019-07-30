'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file TFRealTimeImpFast.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import time
import os
import sys
import numpy as np
import ctypes
import tensorflow as tf

import OpenGL
from OpenGL import GL

FLOAT_SIZE = 4

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import conv_1x1
from MCConvModule import sort_points_step1, sort_points_step2, compute_aabb, \
    poisson_sampling, transform_indexs, compute_pdf, find_neighbors
from PyUtils import save_model

current_milli_time = lambda: time.time() * 1000.0

class AuxPointHierarchy:
    def __init__(self, numLevels):
        self.points_ = []
        self.batchIds_ = []
        for i in range(numLevels):
            self.points_.append(tf.placeholder(tf.float32, [None, 3]))
            self.batchIds_.append(tf.placeholder(tf.int32, [None, 1]))
        self.batchSize_ = 1
        self.hierarchyName_ = "MyAuxHierarchy"    
        with tf.name_scope('MyAuxHierarchy_aabbLayer'):
            aabbMin, aabbMax = compute_aabb(self.points_[0], 
                self.batchIds_[0], 1, True)
        self.aabbMin_ = aabbMin
        self.aabbMax_ = aabbMax  


class AuxOutputPtsValuesPlaceHolders:
    def __init__(self, numFeatures, numInFeatures):
        self.inPoints_ = tf.placeholder(tf.float32, [None, 3])
        self.inBatchIds_ = tf.placeholder(tf.int32, [None, 1])
        self.inFeatures_ = tf.placeholder(tf.float32, [None, numFeatures])
        self.inCellIndexs_ = tf.placeholder(tf.int32, [1, None, None, None, 2])

        self.outPoints_ = tf.placeholder(tf.float32, [None, 3])
        self.outBatchIds_ = tf.placeholder(tf.int32, [None, 1])
        self.outInFeatures_ = tf.placeholder(tf.float32, [None, numInFeatures])

        self.aabbMin_ = tf.placeholder(tf.float32, [1, 3])
        self.aabbMax_ = tf.placeholder(tf.float32, [1, 3]) 

class tfImplementation:

    def __init__(self, pts, features, trainedModel, shaderLoader, 
        gi, sss, envMap,
        radiusConv = 0.05, grow = 8, radiusPD = 0.01):

        self.grow_ = grow

        self.gi_ = gi
        self.sss_ = sss
        
        self.pts_ = pts
        self.features_ = features
        self.radius_ = radiusConv
        self.radiusPD_ = radiusPD
        self.ptsIndexs_ = self.create_initial_poisson_samplings()
        self.batchIds_ = [[[0] for i in range(len(self.ptsIndexs_[0]))],
                [[0] for i in range(len(self.ptsIndexs_[1]))],
                [[0] for i in range(len(self.ptsIndexs_[2]))],
                [[0] for i in range(len(self.ptsIndexs_[3]))]]
        print("Num selected reference points: " +str(len(self.ptsIndexs_[0])))
        print("Num selected reference points: " +str(len(self.ptsIndexs_[1])))
        print("Num selected reference points: " +str(len(self.ptsIndexs_[2])))
        print("Num selected reference points: " +str(len(self.ptsIndexs_[3])))
       
        if self.gi_:
            self.envMap_ = envMap
        
            self.shaderCmp_ = shaderLoader.load_shader(
                ["shaders/directLightCompute.glsl"], [GL.GL_COMPUTE_SHADER])
            self.numVLPUnif_ = GL.glGetUniformLocation(self.shaderCmp_, "numVLP")
            self.numVLPAxisUnif_ = GL.glGetUniformLocation(self.shaderCmp_, "numVLPAxis")
            self.textSMUnif_ = GL.glGetUniformLocation(self.shaderCmp_, "shadowMapTex")
            
            self.ptsSSBO_ = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ptsSSBO_)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.ptsIndexs_[0])*4*FLOAT_SIZE, None, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            
            self.normalsSSBO_ = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.normalsSSBO_)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.ptsIndexs_[0])*4*FLOAT_SIZE, None, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            
            auxDirectLight = np.array([[0.0, 0.0, 0.0, 0.0] for i in range(len(self.ptsIndexs_[0]))]).flatten().tolist()
            ArrayType = (GL.GLfloat*len(auxDirectLight))
            self.directLightSSBO_ = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.directLightSSBO_)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(auxDirectLight)*FLOAT_SIZE, ArrayType(*auxDirectLight), GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        if self.gi_:
            if self.sss_:
                finalNumInFeatures = 9
                numInFeatures = 13
                numIntFeatures = self.grow_
            else:
                finalNumInFeatures = 3
                numInFeatures = 9
                numIntFeatures = self.grow_
        else:
            finalNumInFeatures = 3
            numInFeatures = 3
            numIntFeatures = self.grow_
            
        #Create variable and place holders
        self.auxValsPH_ = AuxOutputPtsValuesPlaceHolders(numIntFeatures, finalNumInFeatures)
        self.auxPtHierarchy_ = AuxPointHierarchy(4)
        self.inFeatures_ = tf.placeholder(tf.float32, [None, numInFeatures])
        self.isTraining_ = tf.placeholder(tf.bool, shape=())

        #Create the network
        sortPts, _, sortFeatures, cellIndexs, aabbMin, aabbMax = \
                    self.create_off_line_network_part(self.auxPtHierarchy_, 
                    self.inFeatures_, numIntFeatures, self.isTraining_)

        mVars = self.get_rt_gi_variables()

        self.sortPts_ = sortPts
        self.sortFeatures_ = sortFeatures
        self.cellIndexs_ = cellIndexs
        self.aabbMin_ = aabbMin
        self.aabbMax_ = aabbMax

        #Create init variables 
        init = tf.global_variables_initializer()
        initLocal = tf.local_variables_initializer()

        #create the saver
        saver = tf.train.Saver()
        
        #Create session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25, visible_device_list='0')
        self.sess_ = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        #Init variables
        self.sess_.run(init, {self.isTraining_: False})
        self.sess_.run(initLocal, {self.isTraining_: False})

        #Restore the model
        saver.restore(self.sess_, trainedModel)

        #Get variables
        weightsRes, weights2Res, weights3Res, biasesRes, biases2Res, biases3Res, bn1moving_meanRes, bn1moving_varianceRes, \
            bn1gammaRes, bn1betaRes, weights_f1Res, biases_f1Res, bn2moving_meanRes, bn2moving_varianceRes, bn2gammaRes, \
            bn2betaRes, weights_f2Res, biases_f2Res = self.sess_.run( \
            [mVars[0], mVars[1], mVars[2], mVars[3], mVars[4], mVars[5], mVars[6], mVars[7], mVars[8], mVars[9], mVars[10],
            mVars[11], mVars[12], mVars[13], mVars[14], mVars[15], mVars[16], mVars[17]])

        self.networkWeights_ = (weightsRes.flatten().tolist(), weights2Res.flatten().tolist(), \
            weights3Res.flatten().tolist(), biasesRes.flatten().tolist(), biases2Res.flatten().tolist(), \
            biases3Res.flatten().tolist(), bn1moving_meanRes.flatten().tolist(), \
            bn1moving_varianceRes.flatten().tolist(), bn1gammaRes.flatten().tolist(), bn1betaRes.flatten().tolist(), \
            weights_f1Res.flatten().tolist(), biases_f1Res.flatten().tolist(), bn2moving_meanRes.flatten().tolist(), \
            bn2moving_varianceRes.flatten().tolist(), bn2gammaRes.flatten().tolist(), bn2betaRes.flatten().tolist(), \
            weights_f2Res.flatten().tolist(), biases_f2Res.flatten().tolist())        
            

    def calculate_abstract_features(self):
        currentPts = self.pts_[0][self.ptsIndexs_[0]]
        currentPts2 = currentPts[self.ptsIndexs_[1]]
        currentPts3 = currentPts2[self.ptsIndexs_[2]]
        currentPts4 = currentPts3[self.ptsIndexs_[3]]
        currentFeatures = self.features_[0][self.ptsIndexs_[0]]

        if self.gi_:
            auxExtraVals = np.full([len(currentPts),1], 1.0, dtype = float)
            auxPts = np.concatenate((currentPts, auxExtraVals), axis=1).flatten().tolist()
            auxNormals = np.concatenate((currentFeatures[:,0:3], auxExtraVals), axis=1).flatten().tolist()
            
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ptsSSBO_)
            vp = GL.glMapBuffer(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_WRITE_ONLY)
            ArrayType = (GL.GLfloat*len(auxPts))
            ctypes.memmove(vp, ArrayType(*auxPts), len(auxPts)*FLOAT_SIZE)
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER) 
            
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.normalsSSBO_)
            vp = GL.glMapBuffer(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_WRITE_ONLY)
            ArrayType = (GL.GLfloat*len(auxNormals))
            ctypes.memmove(vp, ArrayType(*auxNormals), len(auxNormals)*FLOAT_SIZE)
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER) 
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            
            GL.glUseProgram(self.shaderCmp_)
            
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.envMap_.frameBufferShadows_.get_shadow_map())
            GL.glUniform1i(self.textSMUnif_, 0)
            GL.glUniform1i(self.numVLPUnif_, self.envMap_.numVPL_)
            GL.glUniform1i(self.numVLPAxisUnif_, self.envMap_.numPtsAxis_)

            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self.envMap_.vlpPosSSBO_)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, self.envMap_.vlpIntSSBO_)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 5, self.envMap_.vlpViewMatSSBO_)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 6, self.envMap_.vlpProjMatSSBO_)
            
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.ptsSSBO_)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self.normalsSSBO_)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self.directLightSSBO_)
            
            numBlocks = len(self.ptsIndexs_[0])//128
            GL.glDispatchCompute(numBlocks+1, 1, 1)
            GL.glUseProgram(0)
            
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            directLightResults = np.full([len(self.ptsIndexs_[0])*4], 0.0, dtype = float).flatten().tolist()
            ArrayType = (GL.GLfloat*len(directLightResults))
            auxVarDirectLight = ArrayType(*directLightResults)
            
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.directLightSSBO_)
            vp = GL.glMapBuffer(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_READ_ONLY)
            ctypes.memmove(auxVarDirectLight, ctypes.c_void_p(vp), len(directLightResults)*FLOAT_SIZE)
            GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER) 
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            
            directLightResults = np.array(auxVarDirectLight).reshape((-1,4))[:,0:3]

            auxFeatures = np.concatenate((currentFeatures[:,0:3], directLightResults), axis=1)
            currentFeatures = np.concatenate((auxFeatures, currentFeatures[:,3:]), axis=1)
        else:
            currentFeatures = currentFeatures[:,0:3]

        sortPtsRes, sortFeaturesRes, cellIndexsRes, aabbMinRes, aabbMaxRes = \
            self.sess_.run([self.sortPts_, self.sortFeatures_, self.cellIndexs_, self.aabbMin_, self.aabbMax_],
            {self.auxPtHierarchy_.points_[0]: currentPts, self.auxPtHierarchy_.batchIds_[0]: self.batchIds_[0],
            self.auxPtHierarchy_.points_[1]: currentPts2, self.auxPtHierarchy_.batchIds_[1]: self.batchIds_[1],
            self.auxPtHierarchy_.points_[2]: currentPts3, self.auxPtHierarchy_.batchIds_[2]: self.batchIds_[2],
            self.auxPtHierarchy_.points_[3]: currentPts4, self.auxPtHierarchy_.batchIds_[3]: self.batchIds_[3], 
            self.inFeatures_: currentFeatures, self.isTraining_: False})

        return sortPtsRes, sortFeaturesRes, cellIndexsRes, aabbMinRes, aabbMaxRes, currentFeatures


    def create_initial_poisson_samplings(self):

        auxBatchIds = [[0] for i in range(len(self.pts_[0]))]
        
        tf.reset_default_graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, visible_device_list='0')
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        init = tf.global_variables_initializer()
        initLocal = tf.local_variables_initializer()

        sess.run(init)
        sess.run(initLocal)

        currentFeatures = self.features_[0]
        
        inPts = tf.placeholder(tf.float32, [None, 3])
        inBatchIds = tf.placeholder(tf.int32, [None, 1])
        if self.gi_:
            if self.sss_:
                inFeatures = tf.placeholder(tf.float32, [None, 10])
            else:
                inFeatures = tf.placeholder(tf.float32, [None, 6])
        else:
            currentFeatures = currentFeatures[:,0:3]
            inFeatures = tf.placeholder(tf.float32, [None, 3])
        
        mPointHierarchy = PointHierarchy(inPts, inFeatures, inBatchIds, [self.radiusPD_, 0.05, 0.2, 0.8], "MCGI_PH1", 1, relativeRadius = False)
        transformedIndexs1, transformedIndexs2, transformedIndexs3, transformedIndexs4 = \
                sess.run([mPointHierarchy.sampledIndexs_[0], mPointHierarchy.sampledIndexs_[1],
                mPointHierarchy.sampledIndexs_[2], mPointHierarchy.sampledIndexs_[3]],
                {inPts: self.pts_[0], inBatchIds: auxBatchIds, inFeatures: currentFeatures})

        tf.reset_default_graph()

        return [transformedIndexs1, transformedIndexs2, transformedIndexs3, transformedIndexs4]


    def create_off_line_network_part(self, mAuxPointHierarchy, features, k, isTraining):

        def BN_NL_DP_Conv(layerName, inFeatures):
            inFeatures = tf.layers.batch_normalization(inputs = inFeatures, momentum=1.0, 
                trainable = False, training = isTraining, name = layerName+"_BN", renorm=True,
                renorm_clipping=None, renorm_momentum=1.0)
            inFeatures = tf.nn.leaky_relu(inFeatures)
            return inFeatures
            
        numChannels = 1
        if self.gi_:
            numChannels = 3

        mConvBuilder = ConvolutionBuilder(KDEWindow=0.25, relativeRadius = False)

        resultAbstractFeatures = []
        with tf.variable_scope("feature_channel_scope", reuse=tf.AUTO_REUSE):
            for i in range(numChannels):

                numInFeatures = 3
                if self.gi_:
                    numInFeatures = 5
                    inputFeatures = features[:,0:3]
                    inputFeatures = tf.concat([inputFeatures, features[:,3+i:3+i+1]], axis=1)
                    inputFeatures = tf.concat([inputFeatures, features[:,6+i:6+i+1]], axis=1)
                    if self.sss_:
                        numInFeatures = 7
                        inputFeatures = tf.concat([inputFeatures, features[:,9+i:9+i+1]], axis=1)
                        inputFeatures = tf.concat([inputFeatures, features[:,12:13]], axis=1)
                else:
                     inputFeatures = features

                ############################################ Encoder

                # First convolution
                convFeatures1 = mConvBuilder.create_convolution(
                    convName="Conv_1", 
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                    inPointHierarchy=mAuxPointHierarchy,
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
                resultAbstractFeatures.append(bnUpFeatures1)
        
        if self.gi_:
            finalFeatures = tf.concat([resultAbstractFeatures[0], resultAbstractFeatures[1]], axis=1)
            finalFeatures = tf.concat([finalFeatures, resultAbstractFeatures[2]], axis=1)
        else:
            finalFeatures = resultAbstractFeatures[0]

        keys, indexs = sort_points_step1(mAuxPointHierarchy.points_[0], 
            mAuxPointHierarchy.batchIds_[0], mAuxPointHierarchy.aabbMin_, 
            mAuxPointHierarchy.aabbMax_, mAuxPointHierarchy.batchSize_, self.radius_, False)
        sortPts, sortBatchs, sortFeatures, cellIndexs = sort_points_step2(
            mAuxPointHierarchy.points_[0], 
            mAuxPointHierarchy.batchIds_[0], finalFeatures, keys, indexs, 
            mAuxPointHierarchy.aabbMin_, mAuxPointHierarchy.aabbMax_, 
            mAuxPointHierarchy.batchSize_, self.radius_, False)

        return  sortPts, sortBatchs, sortFeatures, cellIndexs, mAuxPointHierarchy.aabbMin_, mAuxPointHierarchy.aabbMax_


    def get_rt_gi_variables(self):
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        
        numOutputs = 1
        numExtraParams = 3
        if self.sss_:
            numExtraParams = 7
        numFeatures = self.grow_
            
        numBlocks = numFeatures/4

        with tf.variable_scope("feature_channel_scope", reuse=tf.AUTO_REUSE):
            
            #Spatial convolution variables.
            weights = tf.get_variable('Final_Conv_weights', [3, 8], initializer=initializer)
            biases = tf.get_variable('Final_Conv_biases', [8], initializer=initializer)
            weights2 = tf.get_variable('Final_Conv_weights2', [2, 4, 4], initializer=initializer)
            weights2 = tf.reshape(weights2, [4, 8])
            biases2 = tf.get_variable('Final_Conv_biases2', [2, 4], initializer=initializer)
            biases2 = tf.reshape(biases2, [8])
            weights3 = tf.get_variable('Final_Conv_weights3', [2, 4, 4], initializer=initializer)
            weights3 = tf.reshape(weights3, [4, 8])
            biases3 = tf.get_variable('Final_Conv_biases3', [2, 4], initializer=initializer)
            biases3 = tf.reshape(biases3, [8])

            #Fully connected layers.
            bn1Mean = tf.get_variable('Final_MLP1_BN_BN/moving_mean', [numFeatures], initializer=initializer)
            bn1Variance = tf.get_variable('Final_MLP1_BN_BN/moving_variance', [numFeatures], initializer=initializer)
            bn1Gamma = tf.get_variable('Final_MLP1_BN_BN/gamma', [numFeatures], initializer=initializer)
            bn1Beta = tf.get_variable('Final_MLP1_BN_BN/beta', [numFeatures], initializer=initializer)
            
            weights_f1 = tf.get_variable('Final_MLP1_weights', [numExtraParams+numFeatures, numFeatures], initializer=initializer)
            biases_f1 = tf.get_variable('Final_MLP1_biases', [numFeatures], initializer=initializer)

            bn2Mean = tf.get_variable('Final_MLP2_BN_BN/moving_mean', [numFeatures], initializer=initializer)
            bn2Variance = tf.get_variable('Final_MLP2_BN_BN/moving_variance', [numFeatures], initializer=initializer)
            bn2Gamma = tf.get_variable('Final_MLP2_BN_BN/gamma', [numFeatures], initializer=initializer)
            bn2Beta = tf.get_variable('Final_MLP2_BN_BN/beta', [numFeatures], initializer=initializer)
                
            # Linear
            weights_f2 = tf.get_variable('Final_MLP2_weights', [numFeatures, numOutputs], initializer=initializer)
            biases_f2 = tf.get_variable('Final_MLP2_biases', [numOutputs], initializer=initializer)

        return (weights, weights2, weights3, biases, biases2, biases3, bn1Mean, bn1Variance, bn1Gamma, bn1Beta, \
            weights_f1, biases_f1, bn2Mean, bn2Variance, bn2Gamma, bn2Beta, weights_f2, biases_f2)



