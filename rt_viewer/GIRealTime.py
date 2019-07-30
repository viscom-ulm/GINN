'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GIRealTime.py

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
import ctypes

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
from OpenGL import GL

import pygame as pg

from OpenGLUtils import ShaderLoader, MeshRenderer, Camera, FrameBuffer, EnvMap
from MeshHelpers import read_model, generate_rendering_buffers, sample_mesh
from TFRealTimeImpFast import tfImplementation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'cuda_ops'))

import screenproc

FLOAT_SIZE = 4
INT_SIZE = 4

current_milli_time = lambda: time.time() * 1000.0

##################################################################### Rendering class and functions


class GLScene:
    
    def __init__(self, width, height, ptMin, ptMax, vertexs, faces, pts, normals, trainedModel, 
        camParams, usePlane, objColor, sssParams, planeColor, convRadius, grow, pdRadius, gi, sss, 
        envMap, lightInt, numVLP):

        self.lighting_ = True

        self.grow_ = grow
        
        # Store gi usage
        self.gi_ = gi or sss
        self.sss_ = sss
        self.lightInt_ = lightInt
        self.useFloor_ = usePlane

        self.planeColor_ = planeColor
        self.objColor_ = objColor
        self.sssParams_ = sssParams
            

        # Configure OpenGL state.
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CW)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)

        # Create the camera.
        self.objCenterPt_ = (ptMax + ptMin)*0.5
        self.objPtMax_ = ptMax
        self.objPtMin_ = ptMin

        aabbSize = math.sqrt(np.sum((ptMax-ptMin) ** 2))
        if camParams is None:
            self.camera_ = Camera(
                    [0.0, 0.0, 0.0], 
                    [0.0, 0.0, -aabbSize*1.5], 
                    [0.0, 1.0, 0.0],
                    float(width)/float(height),
                    45.0, 0.1, aabbSize*5.0)
        else:
            valsCam = []
            with open(camParams, 'r') as paramsFile:
                for line in paramsFile:
                    lineElements = line.split(',')
                    valsCam.append([float(lineElements[0]),float(lineElements[1]),float(lineElements[2])])
            self.camera_ = Camera(
                valsCam[0], 
                valsCam[1], 
                valsCam[2],
                float(width)/float(height),
                45.0, 0.1, aabbSize*5.0)

        self.viewMat_ = self.camera_.get_view_natrix()
        self.projMat_ = self.camera_.get_projection_matrix()

        # Load the shaders.
        self.shaderLoader_ = ShaderLoader()
        self.shaderMesh_ = self.shaderLoader_.load_shader(
            ["shaders/meshVert.glsl", "shaders/meshFrag.glsl"],
            [GL.GL_VERTEX_SHADER, GL.GL_FRAGMENT_SHADER])
        self.viewMatrixUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "viewMatrix")
        self.projMatrixUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "projMatrix")
        self.textSMUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "shadowMapTex")
        self.camPosRenderUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "camPos")
        self.objColorUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "objColor")
        self.numVLPUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "numVLP")
        self.numVLPAxisUnif_ = GL.glGetUniformLocation(self.shaderMesh_, "numVLPAxis")

        self.shaderRender_ = self.shaderLoader_.load_shader(
            ["shaders/renderVert.glsl", "shaders/renderFrag.glsl"],
            [GL.GL_VERTEX_SHADER, GL.GL_FRAGMENT_SHADER])
        self.textUnif_ = GL.glGetUniformLocation(self.shaderRender_, "colorTex")
        self.textPosUnif_ = GL.glGetUniformLocation(self.shaderRender_, "posTex")
        self.textNormalUnif_ = GL.glGetUniformLocation(self.shaderRender_, "normalTex")
        self.textMattUnif_ = GL.glGetUniformLocation(self.shaderRender_, "matTex")
        self.textLightUnif_ = GL.glGetUniformLocation(self.shaderRender_, "lightTex")
        self.directLightRenderUnif_ = GL.glGetUniformLocation(self.shaderRender_, "directLight")
        self.giRenderUnif_ = GL.glGetUniformLocation(self.shaderRender_, "gi")

        # Load the mesh.
        self.mesh_ = MeshRenderer(np.array(vertexs), np.array(faces), [3,3])
        
        # Add color to the normals for GI.
        if self.gi_:
            auxBuff = np.array([[self.objColor_[0], self.objColor_[1], self.objColor_[2]] for i in range(len(pts[0]))])
            finalNormals = []
            for i in range(len(pts)):
                finalNormals.append(np.concatenate((normals[i], auxBuff), axis=1))
            normals = np.array(finalNormals)

        #Add material properties for sss.
        if self.sss_:
            auxBuff = np.array([self.sssParams_ for i in range(len(pts[0]))])
            finalNormals = []
            for i in range(len(pts)):
                finalNormals.append(np.concatenate((normals[i], auxBuff), axis=1))
            normals = np.array(finalNormals)
        
        # Create the floor
        if self.useFloor_:
            self.floor_ = MeshRenderer(np.array([[-aabbSize*0.4,  self.objPtMin_[1]-self.objCenterPt_[1]-0.005, -aabbSize*0.4, 0.0, 1.0, 0.0,
                                                    aabbSize*0.4,  self.objPtMin_[1]-self.objCenterPt_[1]-0.005, -aabbSize*0.4, 0.0, 1.0, 0.0,
                                                    aabbSize*0.4, self.objPtMin_[1]-self.objCenterPt_[1]-0.005, aabbSize*0.4, 0.0, 1.0, 0.0,
                                                    -aabbSize*0.4, self.objPtMin_[1]-self.objCenterPt_[1]-0.005, aabbSize*0.4, 0.0, 1.0, 0.0]]),
                                                    np.array([0, 1, 2, 0, 2, 3]),
                                                    [3,3])
            if not self.sss_:
                planePts = []    
                planeNormals = []
                for i in range(50000):
                    planePts.append([(np.random.random()- 0.5)*aabbSize*0.8, self.objPtMin_[1]-self.objCenterPt_[1], (np.random.random()- 0.5)*aabbSize*0.8])
                    if self.gi_:
                        planeNormals.append([0.0, 1.0, 0.0, self.planeColor_[0], self.planeColor_[1], self.planeColor_[2]])
                    else:
                        planeNormals.append([0.0, 1.0, 0.0])

                planePts = np.array(planePts)
                planeNormals = np.array(planeNormals)
                finalPts = []
                finalNormals = []
                for i in range(len(pts)):
                    finalPts.append(np.concatenate((pts[i], planePts), axis=0))
                    finalNormals.append(np.concatenate((normals[i], planeNormals), axis=0))
                pts = np.array(finalPts)
                normals = np.array(finalNormals)


        # Create the quad mesh.
        self.quad_ = MeshRenderer(np.array([[-1.0,  1.0, 0.5, 
                                    1.0,  1.0, 0.5,
                                    1.0, -1.0, 0.5,
                                   -1.0, -1.0, 0.5]]),
                                  np.array([0, 1, 2, 0, 2, 3]),
                                  [3])

        # Create environment map.
        self.envMap_ = EnvMap(envMap, self.shaderLoader_, aabbSize, lightInt, 8192, numVLP)
        self.envMap_.clear_shadow_map()
        self.envMap_.update_shadow_maps(self.mesh_)
        if self.useFloor_:
            self.envMap_.update_shadow_maps(self.floor_)

        # Create the frame buffer.
        self.frameBuffer_ = FrameBuffer([GL.GL_RGBA32F, GL.GL_RGBA32F, GL.GL_RGBA32F, GL.GL_RGBA32F], width, height)
        
        # Create output texture
        self.outTexture_ = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.outTexture_)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, width, height, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Resize viewport.
        self.width_ = width
        self.height_ = height
        GL.glViewport(0, 0, width, height)

        # Initialize mouse variables.
        self.lastRotated_ = False
        self.mouseX_ = 0.0
        self.mouseY_ = 0.0

        self.tfImplementation_ = tfImplementation(pts, normals, trainedModel,  
            self.shaderLoader_, self.gi_, self.sss_, self.envMap_, 
            convRadius, grow, pdRadius)
        self.init_tf_weights_buffers()
        self.init_abstract_feature_pts_buffers()

        self.accumTime_ = 0.0
        self.accumTimeCounter_ = 0


    def init_abstract_feature_pts_buffers(self):

        pts, features, cellIndexs, aabbMin, aabbMax, _ = self.tfImplementation_.calculate_abstract_features()
        self.gridSize_ = cellIndexs.shape[1]
        pts = pts.flatten().tolist()
        features = features.flatten().tolist()
        cellIndexs = cellIndexs.flatten().tolist()
        aabb = np.concatenate((aabbMin, aabbMax), axis=0).flatten().tolist()

        self.ptsSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ptsSSBO_)
        ArrayType = GL.GLfloat*len(pts)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(pts)*FLOAT_SIZE, ArrayType(*pts), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.featuresSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.featuresSSBO_)
        ArrayType = GL.GLfloat*len(features)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(features)*FLOAT_SIZE, ArrayType(*features), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.cellIndexsSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.cellIndexsSSBO_)
        ArrayType = GL.GLuint*len(cellIndexs)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(cellIndexs)*INT_SIZE, ArrayType(*cellIndexs), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.aabbSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.aabbSSBO_)
        ArrayType = GL.GLfloat*len(aabb)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(aabb)*FLOAT_SIZE, ArrayType(*aabb), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        mTexture1 = self.frameBuffer_.get_texture(0)
        mTexture2 = self.frameBuffer_.get_texture(1)
        screenproc.initInteroperabilityGLCUDA(self.width_, self.height_, int(mTexture1),
            int(mTexture2), int(self.outTexture_), int(self.ptsSSBO_), int(self.featuresSSBO_), int(self.aabbSSBO_), 
            int(self.cellIndexsSSBO_), self.tfImplementation_.radius_)
            
        if self.gi_:
            screenproc.initInteroperabilityGLCUDAGI()
        if self.sss_:
            screenproc.initInteroperabilityGLCUDASSS(self.frameBuffer_.get_texture(2), self.frameBuffer_.get_texture(3), 
                self.sssParams_[0], self.sssParams_[1], self.sssParams_[2], self.sssParams_[3])


    def init_tf_weights_buffers(self):

        self.weightsConv1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.weightsConv1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[0])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[0])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[0]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.weightsConv2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.weightsConv2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[1])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[1])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[1]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.weightsConv3_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.weightsConv3_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[2])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[2])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[2]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.biasesConv1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.biasesConv1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[3])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[3])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[3]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.biasesConv2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.biasesConv2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[4])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[4])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[4]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.biasesConv3_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.biasesConv3_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[5])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[5])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[5]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.meanBN1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.meanBN1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[6])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[6])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[6]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.varianceBN1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.varianceBN1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[7])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[7])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[7]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.gammaBN1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.gammaBN1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[8])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[8])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[8]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.betaBN1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.betaBN1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[9])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[9])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[9]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.weightsMLP1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.weightsMLP1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[10])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[10])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[10]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.biasesMLP1_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.biasesMLP1_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[11])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[11])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[11]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.meanBN2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.meanBN2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[12])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[12])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[12]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.varianceBN2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.varianceBN2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[13])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[13])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[13]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.gammaBN2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.gammaBN2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[14])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[14])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[14]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.betaBN2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.betaBN2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[15])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[15])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[15]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.weightsMLP2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.weightsMLP2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[16])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[16])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[16]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.biasesMLP2_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.biasesMLP2_)
        ArrayType = GL.GLfloat*len(self.tfImplementation_.networkWeights_[17])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(self.tfImplementation_.networkWeights_[17])*FLOAT_SIZE, 
            ArrayType(*self.tfImplementation_.networkWeights_[17]), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        screenproc.initInteroperabilityGLCUDAConvWeights(self.grow_,
            int(self.weightsConv1_), int(self.weightsConv2_), int(self.weightsConv3_), 
            int(self.biasesConv1_), int(self.biasesConv2_), int(self.biasesConv3_))
        screenproc.initInteroperabilityGLCUDAMLPWeights(
            int(self.meanBN1_), int(self.varianceBN1_), int(self.gammaBN1_), int(self.betaBN1_), 
            int(self.weightsMLP1_), int(self.biasesMLP1_), 
            int(self.meanBN2_), int(self.varianceBN2_) , int(self.gammaBN2_), int(self.betaBN2_), 
            int(self.weightsMLP2_), int(self.biasesMLP2_))



    def update(self, rotate, mouseX, mouseY):
        if rotate and self.lastRotated_:
            self.camera_.rotate_x((mouseY-self.mouseY_)/500.0)
            self.camera_.rotate_y((mouseX-self.mouseX_)/500.0)
            self.viewMat_ = self.camera_.get_view_natrix()
        self.mouseX_ = mouseX
        self.mouseY_ = mouseY
        self.lastRotated_ = rotate


    def display(self):
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(100,100,100,100)

        #Render G-buffer
        self.frameBuffer_.bind()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)     
        GL.glUseProgram(self.shaderMesh_)
        GL.glBindFragDataLocation(self.shaderMesh_, 0, "outputPos")
        GL.glBindFragDataLocation(self.shaderMesh_, 1, "outputNormal")
        GL.glBindFragDataLocation(self.shaderMesh_, 2, "outputMat")
        GL.glBindFragDataLocation(self.shaderMesh_, 3, "outputLight")
        GL.glUniformMatrix4fv(self.viewMatrixUnif_, 1, GL.GL_TRUE, np.ascontiguousarray(self.viewMat_, dtype=np.float32))
        GL.glUniformMatrix4fv(self.projMatrixUnif_, 1, GL.GL_TRUE, np.ascontiguousarray(self.projMat_, dtype=np.float32))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.envMap_.frameBufferShadows_.get_shadow_map())
        GL.glUniform1i(self.textSMUnif_, 0)
        
        GL.glUniform3f(self.camPosRenderUnif_, self.camera_.obs_[0], self.camera_.obs_[1], self.camera_.obs_[2])
        GL.glUniform1i(self.numVLPUnif_, self.envMap_.numVPL_)
        GL.glUniform1i(self.numVLPAxisUnif_, self.envMap_.numPtsAxis_)

        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.envMap_.vlpPosSSBO_)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self.envMap_.vlpIntSSBO_)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self.envMap_.vlpViewMatSSBO_)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, self.envMap_.vlpProjMatSSBO_)
        
        if self.useFloor_:
            GL.glUniform3f(self.objColorUnif_, self.planeColor_[0], self.planeColor_[1], self.planeColor_[2])
            self.floor_.render_mesh()
            
        GL.glUniform3f(self.objColorUnif_, self.objColor_[0], self.objColor_[1], self.objColor_[2])
        self.mesh_.render_mesh()   

        GL.glUseProgram(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        #Compute NN
        screenproc.computeAOTexture(self.gridSize_)

        #Render result.
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT) 
        attachments = [GL.GL_COLOR_ATTACHMENT0]
        GL.glDrawBuffers(1,  attachments)
        GL.glUseProgram(self.shaderRender_)
        GL.glBindFragDataLocation(self.shaderRender_, 0, "outputColor")

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.outTexture_)
        GL.glUniform1i(self.textUnif_, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(0))
        GL.glUniform1i(self.textPosUnif_, 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(1))
        GL.glUniform1i(self.textNormalUnif_, 2)

        GL.glActiveTexture(GL.GL_TEXTURE3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.frameBuffer_.get_texture(2))
        GL.glUniform1i(self.textMattUnif_, 3)

        GL.glActiveTexture(GL.GL_TEXTURE4)
        GL.glBindTexture(GL.GL_TEXTURE_2D,self.frameBuffer_.get_texture(3))
        GL.glUniform1i(self.textLightUnif_, 4)

        GL.glActiveTexture(GL.GL_TEXTURE0)

        if self.lighting_:
            GL.glUniform1f(self.directLightRenderUnif_, 0.0)
        else:
            GL.glUniform1f(self.directLightRenderUnif_, 1.0)
            
        if self.gi_:
            if self.sss_:
                GL.glUniform2f(self.giRenderUnif_, 1.0, 1.0)
            else:
                GL.glUniform2f(self.giRenderUnif_, 1.0, 0.0)
        else:
            GL.glUniform2f(self.giRenderUnif_, 0.0, 0.0)

        self.quad_.render_mesh()
        GL.glUseProgram(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

##################################################################### MAIN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train GINN')
    parser.add_argument('--in3DModel', help='3D input model')
    parser.add_argument('--in3DModelScale', default=1.0, type=float, help='3DModelScale (default: 1.0)')
    parser.add_argument('--inTrainedModel', default='../trained_networks/sss.ckpt', help='Input trained model (default: ../trained_networks/sss.ckpt)')
    parser.add_argument('--model', default='MCGINetwork', help='model (default: MCGINetwork)')
    parser.add_argument('--grow', default=8, type=int, help='Grow rate (default: 8)')
    parser.add_argument('--camParams', default=None, help='Camera parameters file')
    parser.add_argument('--usePlane', action='store_true', help='Use ground plane (default: False)')
    parser.add_argument('--convRadius', default=0.05, type=float, help='Radius convolution (default: 0.05)')
    parser.add_argument('--pdRadius', default=0.01, type=float, help='Radius poisson disk (default: 0.01)')
    parser.add_argument('--lightIntensity', default=100.0, type=float, help='Light intensity (default: 100.0)')
    parser.add_argument('--envMap', default='env_maps/spruit_sunrise_1k.hdr', help='model (default: envmaps/spruit_sunrise_1k.hdr)')
    parser.add_argument('--numVLP', default=1024, type=int, help='Number of virtual point lights (default: 1024)')
    parser.add_argument('--objColor', default=[1.0, 1.0, 1.0], type=float, nargs=3,  help='Object color (default: [1.0, 1.0, 1.0])')
    parser.add_argument('--sssParams', default=[1.0, 1.0, 1.0, 1.0], type=float, nargs=4,  help='SSS Params (default: [1.0, 1.0, 1.0, 1.0])')    
    parser.add_argument('--planeColor', default=[1.0, 0.05, 0.05], type=float, nargs=3,  help='Plane color (default: [1.0, 0.0, 0.0])')
    parser.add_argument('--gi', action='store_true', help='Use global illumination (default: False)')
    parser.add_argument('--sss', action='store_true', help='Use global illumination (default: False)')
    args = parser.parse_args()    

    #Initialize random seed
    np.random.seed(int(time.time()))

    # #Load the model
    vertexs, normals, faces, coordMin, coordMax = read_model(args.in3DModel, 0, args.in3DModelScale)
    rendVert, rendFaces = generate_rendering_buffers(vertexs, normals, faces)
    pts, ptsNormals = sample_mesh(vertexs, normals, faces, 100000)
    
    print("Loaded model "+args.in3DModel+" and sampled "+str(len(pts))+" from it.")
    print("Vertexs: "+str(len(rendVert[0])))
    print("faces: "+str(len(rendFaces)))
    print (coordMin)
    print (coordMax)

    #Render
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    SCREEN = pg.display.set_mode((1024, 1024),  pg.OPENGL | pg.DOUBLEBUF)
    MyClock = pg.time.Clock()

    MyGL = GLScene(1024, 1024, coordMin, coordMax, rendVert, rendFaces, 
        pts, ptsNormals, args.inTrainedModel, 
        args.camParams, args.usePlane, 
        args.objColor, args.sssParams, args.planeColor, 
        args.convRadius, args.grow, args.pdRadius, args.gi, args.sss,
        args.envMap, args.lightIntensity, args.numVLP)

    mouseDown = False
    while 1:
        for event in pg.event.get():
            if event.type==pg.QUIT or (event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE):
                pg.quit();sys.exit()
            elif (event.type==pg.KEYDOWN and event.key==pg.K_s):
                if not os.path.exists("export"): os.mkdir("export")
                realFileName = os.path.basename(args.in3DModel)
                savedFile = False
                auxIterator = 0
                while not savedFile:
                    if not os.path.exists("export/"+realFileName+"_"+str(auxIterator)+".png"):
                        pg.image.save(SCREEN,"export/"+realFileName+"_"+str(auxIterator)+".png")
                        savedFile = True                        
                    else:
                        auxIterator += 1
            elif (event.type==pg.KEYDOWN and event.key==pg.K_l):
                MyGL.lighting_ = not MyGL.lighting_
            elif event.type == pg.KEYDOWN:
                pass
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouseDown = True
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    mouseDown = False
        
        mouseX, mouseY = pg.mouse.get_pos()
        MyGL.update(mouseDown, mouseX, mouseY)

        MyGL.display()

        pg.display.flip()