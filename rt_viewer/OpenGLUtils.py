'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file OpenGLUtils.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import os
import numpy as np
import ctypes

import pygame as pg

import OpenGL
from OpenGL import GL

import imageio

FLOAT_SIZE = 4
INT_SIZE = 4

class ShaderLoader:
    
    def __init__(self):
        pass

    def _link_(self, program):
        GL.glLinkProgram(program)

        status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
        if not status:
            log = GL.glGetProgramInfoLog(program)
            raise RuntimeError("Linking failue: "+log.decode('UTF-8'))

    def _compile_(self, shaderPath, shaderCode, shaderType):
        shader = GL.glCreateShader(shaderType)

        GL.glShaderSource(shader, shaderCode)

        GL.glCompileShader(shader)

        status = GL.glGetShaderiv(shader,GL.GL_COMPILE_STATUS)
        if not status:
            log = GL.glGetShaderInfoLog(shader)
            raise RuntimeError("Compile failure in shader: "+shaderPath+ "\n "+log.decode('UTF-8'))

        return shader

    def _load_shader_code_(self, shaderPath):
        shaderCode = ""
        with open(shaderPath, 'r') as modelFile:        
            for line in modelFile:
                shaderCode += line
        return shaderCode
    
    def load_shader(self, shaderPathList, shaderTypes):
        currProgram = GL.glCreateProgram()

        shadeList = []
        for shaderPath, shaderType in zip(shaderPathList, shaderTypes):
            shadeList.append(self._compile_(shaderPath, 
                self._load_shader_code_(shaderPath), shaderType))
        
        for shade in shadeList:
            GL.glAttachShader(currProgram,shade)
        
        self._link_(currProgram)

        for shade in shadeList:
            GL.glDetachShader(currProgram,shade)
            GL.glDeleteShader(shade)

        return currProgram

    
class MeshRenderer:
    
    def __init__(self, verts, trians, elements):
        self.verts_ = verts
        self.trians_ = trians
        self.elements_ = elements
        self._create_buffers_()
        self._init_vao_()

    def _create_buffers_(self):
        flattenVerts = self.verts_[0].tolist()
        self.vbo_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_)
        ArrayType = (GL.GLfloat*len(flattenVerts))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(flattenVerts)*FLOAT_SIZE,
                        ArrayType(*flattenVerts), GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,0)

        flattenIndexs = self.trians_.tolist()
        self.ibo_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo_)
        ArrayType = (GL.GLuint*len(flattenIndexs))
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(flattenIndexs)*INT_SIZE,
                        ArrayType(*flattenIndexs), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,0)

    def _init_vao_(self):
        self.vao_ = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao_)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_)
        vertexSize = 0
        for currElem in self.elements_:
            vertexSize += currElem
        accumVertexSize = 0
        for it, currElem in enumerate(self.elements_):
            GL.glEnableVertexAttribArray(it)
            GL.glVertexAttribPointer(it, currElem, GL.GL_FLOAT, GL.GL_FALSE, vertexSize*FLOAT_SIZE, GL.GLvoidp(accumVertexSize*FLOAT_SIZE))
            accumVertexSize += currElem
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo_)
        GL.glBindVertexArray(0)
        
    def update_buffer(self, index):
        flattenVerts = self.verts_[index].tolist()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_)
        vp = GL.glMapBuffer(GL.GL_ARRAY_BUFFER, GL.GL_WRITE_ONLY)
        ArrayType = (GL.GLfloat*len(flattenVerts))
        ctypes.memmove(vp, ArrayType(*flattenVerts), len(flattenVerts)*FLOAT_SIZE)
        GL.glUnmapBuffer(GL.GL_ARRAY_BUFFER) 
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        

    def render_mesh(self):
        GL.glBindVertexArray(self.vao_)
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.trians_), GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)


class Camera:

    def __init__(self, vrp, obs, upVec, ar, fov, zNear, zFar, 
        perspective = True, right=0.0, top=0.0):
        self.vrp_ = np.array(vrp)
        self.obs_ = np.array(obs)
        self.upVec_ = np.array(upVec)
        self.ar_ = ar
        self.fov_ = fov
        self.zNear_ = zNear
        self.zFar_ = zFar
        self.prespective_ = perspective
        self.right_ = right
        self.top_ = top


    def _normalize_(self, v):
        m = math.sqrt(np.sum(v ** 2))
        if m == 0:
            return v
        return v / m


    def rotate_y(self, angle):
        cosVal = np.cos(angle)
        sinVal = np.sin(angle)
        T = np.array([[cosVal, 0.0, -sinVal],
                       [0.0, 1.0, 0.0],
                       [sinVal, 0.0, cosVal]])
        auxPos = self.obs_ - self.vrp_
        auxPos = np.dot(T, auxPos)[:3]
        self.obs_ = auxPos + self.vrp_


    def rotate_x(self, angle):
        F = self.vrp_ - self.obs_
        f = self._normalize_(F)
        U = self._normalize_(self.upVec_)
        axis = np.cross(f, U)
        
        x, y, z = self._normalize_(axis)
        s = math.sin(-angle)
        c = math.cos(-angle)
        nc = 1 - c
        T = np.array([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s],
                        [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s],
                        [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c]])
                        

        auxPos = self.obs_ - self.vrp_
        auxPos = np.dot(T, auxPos)
        self.obs_ = auxPos + self.vrp_


    def get_view_natrix(self):
        F = self.vrp_ - self.obs_
        f = self._normalize_(F)
        U = self._normalize_(self.upVec_)
        s = self._normalize_(np.cross(f, U))
        u = self._normalize_(np.cross(s, f))
        M = np.matrix(np.identity(4))
        M[:3,:3] = np.vstack([s,u,-f])
        T = np.matrix([[1.0, 0.0, 0.0, -self.obs_[0]],
                       [0.0, 1.0, 0.0, -self.obs_[1]],
                       [0.0, 0.0, 1.0, -self.obs_[2]],
                       [0.0, 0.0, 0.0, 1.0]])
        return  M * T


    def get_projection_matrix(self):
        if self.prespective_:
            s = 1.0/math.tan(math.radians(self.fov_)/2.0)
            sx, sy = s / self.ar_, s
            zz = (self.zFar_+self.zNear_)/(self.zNear_-self.zFar_)
            zw = (2*self.zFar_*self.zNear_)/(self.zNear_-self.zFar_)
            return np.matrix([[sx,0,0,0],
                            [0,sy,0,0],
                            [0,0,zz,zw],
                            [0,0,-1,0]])
        else:
            zw = (self.zFar_+self.zNear_)/(self.zNear_-self.zFar_)
            zz = 2.0/(self.zNear_-self.zFar_)
            return np.matrix([[1.0/self.right_,0,0,0],
                            [0,1.0/self.top_,0,0],
                            [0,0,zz,zw],
                            [0,0,0,1]])


class FrameBufferShadowMap:

    def __init__(self, width, height):
        # Create the frame buffer.
        self.fbo_ = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)

         # Create the textures of the frame buffer.
        self.texture_ = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_REF_TO_TEXTURE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT32F, width, height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, self.texture_, 0)
        GL.glDrawBuffer(GL.GL_NONE)

        # Check if the frame buffer was created properly.
        if not GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Bind frame buffer failed")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)
        GL.glDrawBuffer(GL.GL_NONE)

    def get_shadow_map(self):
        return self.texture_


class FrameBuffer:

    def __init__(self, bufferFormats, width, height):
        if len(bufferFormats) > 6:
            raise RuntimeError("Number of attachement to buffer too high: "+str(len(bufferFormats)))

        # Create the frame buffer.
        self.textures_ = []
        self.fbo_ = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)
        
        self.colorAttList_ = [GL.GL_COLOR_ATTACHMENT0,
                              GL.GL_COLOR_ATTACHMENT1,
                              GL.GL_COLOR_ATTACHMENT2,
                              GL.GL_COLOR_ATTACHMENT3,
                              GL.GL_COLOR_ATTACHMENT4,
                              GL.GL_COLOR_ATTACHMENT5,
                              GL.GL_COLOR_ATTACHMENT6]
        self.formatDict_ = {
                GL.GL_RGBA32F: (GL.GL_RGBA, GL.GL_FLOAT),
                GL.GL_RGBA: (GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)}

         # Create the textures of the frame buffer.
        for it, currFormat in enumerate(bufferFormats):
            if not(currFormat in self.formatDict_):
                raise RuntimeError("The texture format is not in the dictionary: "+str(currFormat))
            myFormat = self.formatDict_[currFormat]
            texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, currFormat, width, height, 0, myFormat[0], myFormat[1], None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, self.colorAttList_[it], GL.GL_TEXTURE_2D, texture, 0)
            self.textures_.append(texture)

        # Creaet the render buffer.
        self.rbo_ = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.rbo_)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, self.rbo_)

        # Check if the frame buffer was created properly.
        if not GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Bind frame buffer failed")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo_)
        GL.glDrawBuffers(len(self.textures_),  self.colorAttList_)

    def get_texture(self, index):
        return self.textures_[index]


class EnvMap:

    def __init__(self, envMapPath, shaderLoader, aabbSize, intensity, fbSize= 8192, numVPL=4):
        self.numVPL_ = numVPL
        self.envMapPath_ = envMapPath
        self.texid_ = 0
        self.shaderLoader_ = shaderLoader
        self.fbSize_ = fbSize
        self.intensity_ = intensity
        self.numPtsAxis_ = int(math.ceil(math.sqrt(numVPL)))
        self.vlpFBSize_ = self.fbSize_//self.numPtsAxis_

        self._create_frame_buffer()
        self._load_shaders()
        self._load_texture()
        self._create_VPL(aabbSize)
        self._create_SSBO_()


    def _create_frame_buffer(self):
        self.frameBufferShadows_ = FrameBufferShadowMap(self.fbSize_, self.fbSize_)


    def _load_shaders(self):
        self.shaderRenderShadows_ = self.shaderLoader_.load_shader(
            ["shaders/renderVertShadows.glsl", "shaders/renderFragShadows.glsl"],
            [GL.GL_VERTEX_SHADER, GL.GL_FRAGMENT_SHADER])
        self.viewMatrixUnifShadows_ = GL.glGetUniformLocation(self.shaderRenderShadows_, "viewMatrix")
        self.projMatrixUnifShadows_ = GL.glGetUniformLocation(self.shaderRenderShadows_, "projMatrix")


    def _load_texture(self):
        self.tex_ = imageio.imread(self.envMapPath_)*self.intensity_

        self.texWidth_ = float(self.tex_.shape[1])
        self.texHeight_ = float(self.tex_.shape[0])   

    def save_texture_pfm(self, path):
        imageio.imwrite(path, self.tex_/self.intensity_)

    def _create_VPL(self, aabbSize):

        luminance = self.tex_.copy()
        luminance[:,:,0] = luminance[:,:,0]*0.2126
        luminance[:,:,1] = luminance[:,:,1]*0.7152
        luminance[:,:,2] = luminance[:,:,2]*0.0722
        luminance = np.sum(luminance, axis=2)
        for curH in range(int(self.texHeight_)):
            curHF = (float(curH))/self.texHeight_
            sinTheta = math.sin(math.pi * curHF)
            for curW in range(int(self.texWidth_)):
                luminance[curH, curW] = luminance[curH, curW]*sinTheta

        conditional = np.zeros((luminance.shape[0], luminance.shape[1]+1))
        conditional[:, 0] = 0.0
        for auxJ in range(luminance.shape[1]):
            conditional[:, auxJ+1] = conditional[:, auxJ] + luminance[:,auxJ]/float(luminance.shape[1])
        maxConditional = conditional[:,-1].copy()
        conditional = conditional/np.tile(conditional[:,-1].reshape((-1, 1)), (1, conditional.shape[1]))

        marginal = np.zeros(maxConditional.shape[0]+1)
        marginal[0] = 0.0
        for auxJ in range(maxConditional.shape[0]):
            marginal[auxJ+1] = marginal[auxJ] + maxConditional[auxJ]/float(maxConditional.shape[0])
        maxMarginal = marginal[-1]
        marginal = marginal/marginal[-1]


        self.vlpPos_ = []
        self.vlpIntensities_ = []
        self.vlpViewMats_ = []
        self.vlpProjMats_ = []        
        auxCam = Camera(
                    [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                    1.0, 3.0, 3.0-aabbSize*0.6, 3.0+aabbSize*0.6,
                    False, aabbSize*0.5, aabbSize*0.5)

        np.random.seed(0)
        for curPtIt in range(self.numVPL_):
            ptUV = np.random.uniform(size=(2))
            
            vIndex = np.digitize(ptUV[1], marginal)-1
            dv = (ptUV[1] - marginal[vIndex])/(marginal[vIndex+1]-marginal[vIndex])
            d1 = min(int(math.floor(float(vIndex)+ dv + 0.5)), self.tex_.shape[1]-1)
            pdfv = maxConditional[vIndex]/maxMarginal

            uIndex = np.digitize(ptUV[0], conditional[vIndex])-1
            du = (ptUV[0] - conditional[vIndex, uIndex])/(conditional[vIndex, uIndex+1]-conditional[vIndex, uIndex])
            d0 = min(int(math.floor(float(uIndex) + du + 0.5)), self.tex_.shape[0]-1)
            pdfu = luminance[vIndex, uIndex]/maxConditional[vIndex]

            currPDF = pdfv*pdfu

            theta = (float(d1)/float(marginal.shape[0])) * np.pi
            phi = (float(d0)/float(conditional.shape[1])) * 2.0 * np.pi
            cosTheta = math.cos(theta)
            sinTheta = math.sin(theta)
            sinPhi = math.sin(phi)
            cosPhi = math.cos(phi)
            curPt = np.array([sinTheta*cosPhi, cosTheta, sinTheta*sinPhi])
            self.vlpPos_.append(curPt)
            currPDF = currPDF / (2.0 * np.pi * np.pi * sinTheta)


            currValue = np.array(self.tex_[d1, d0])
            self.vlpIntensities_.append(np.array([currValue[0], currValue[1], currValue[2], currPDF]))
            auxCam.obs_ = curPt*3.0
            self.vlpViewMats_.append(auxCam.get_view_natrix())
            self.vlpProjMats_.append(auxCam.get_projection_matrix())
           

    
    def _create_SSBO_(self):
        auxOneArray = np.full((self.numVPL_, 1), 1.0, dtype=float)
        auxVLPPos = np.concatenate((self.vlpPos_, auxOneArray), axis=1)
        vlpPosFlatten = auxVLPPos.flatten().tolist()

        self.vlpPosSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.vlpPosSSBO_)
        ArrayType = GL.GLfloat*len(vlpPosFlatten)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(vlpPosFlatten)*FLOAT_SIZE, ArrayType(*vlpPosFlatten), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        vlpIntensityFlatten = np.array(self.vlpIntensities_).flatten().tolist()

        self.vlpIntSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.vlpIntSSBO_)
        ArrayType = GL.GLfloat*len(vlpIntensityFlatten)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(vlpIntensityFlatten)*FLOAT_SIZE, ArrayType(*vlpIntensityFlatten), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        auxViewMats = []
        auxProjMats = []
        for vlpIter in range(len(self.vlpViewMats_)):
            auxViewMats = auxViewMats + np.ascontiguousarray(self.vlpViewMats_[vlpIter]).flatten().tolist()
            auxMatrix = np.matrix([
                    [0.5, 0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 1.0]])
            auxMatrix = np.dot(auxMatrix, self.vlpProjMats_[vlpIter])
            auxProjMats = auxProjMats + np.ascontiguousarray(auxMatrix).flatten().tolist()
        
        self.vlpViewMatSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.vlpViewMatSSBO_)
        ArrayType = GL.GLfloat*len(auxViewMats)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(auxViewMats)*FLOAT_SIZE, ArrayType(*auxViewMats), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

        self.vlpProjMatSSBO_ = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.vlpProjMatSSBO_)
        ArrayType = GL.GLfloat*len(auxProjMats)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(auxProjMats)*FLOAT_SIZE, ArrayType(*auxProjMats), GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)


    def clear_shadow_map(self):
        self.frameBufferShadows_.bind()
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)


    def update_shadow_maps(self, mesh):
        self.frameBufferShadows_.bind()

        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)  

        GL.glUseProgram(self.shaderRenderShadows_)
        GL.glBindFragDataLocation(self.shaderRenderShadows_, 0, "outputPos")
        
        for vlpIt, curPt in enumerate(self.vlpPos_):
            xCoord = vlpIt//self.numPtsAxis_
            yCoord = vlpIt%self.numPtsAxis_

            GL.glViewport(xCoord*self.vlpFBSize_, yCoord*self.vlpFBSize_, self.vlpFBSize_, self.vlpFBSize_)
            GL.glUniformMatrix4fv(self.viewMatrixUnifShadows_, 1, GL.GL_TRUE, np.ascontiguousarray(self.vlpViewMats_[vlpIt], dtype=np.float32))
            GL.glUniformMatrix4fv(self.projMatrixUnifShadows_, 1, GL.GL_TRUE, np.ascontiguousarray(self.vlpProjMats_[vlpIt], dtype=np.float32))
            mesh.render_mesh()    

        GL.glUseProgram(0)