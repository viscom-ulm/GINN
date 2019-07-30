'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MeshHelpers.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import math
import os
import numpy as np

def read_model(myFile, numPoses=1, scale=1.0, invertZ = True):
    invert = 1.0
    if invertZ:
        invert = -1.0
    vertexs = []
    normals = []
    faces = []

    dirName = os.path.dirname(myFile)

    if numPoses == 0:
        currVertexs = []
        currNormals = []
        with open(myFile+".obj", 'r') as modelFile:    
            for line in modelFile:
                lineElements = line.split()
                if len(lineElements) > 0:
                    if lineElements[0] == "v":
                        currVertexs.append([float(lineElements[1]), float(lineElements[2]), invert*float(lineElements[3])])
                    elif lineElements[0] == "vn":
                        auxNormal = [float(lineElements[1]), float(lineElements[2]), invert*float(lineElements[3])]
                        normalLength = math.sqrt(auxNormal[0]*auxNormal[0] + auxNormal[1]*auxNormal[1] + auxNormal[2]*auxNormal[2])
                        if normalLength > 0.0:
                            auxNormal[0] = auxNormal[0]/normalLength
                            auxNormal[1] = auxNormal[1]/normalLength
                            auxNormal[2] = auxNormal[2]/normalLength
                        currNormals.append([auxNormal[0], auxNormal[1], auxNormal[2]])
                    elif lineElements[0] == "f":
                        vertex1 = lineElements[1].split('/')
                        vertex2 = lineElements[2].split('/')
                        vertex3 = lineElements[3].split('/')
                        faces.append([[int(vertex1[0])-1, int(vertex1[2])-1], [int(vertex2[0])-1, int(vertex2[2])-1], [int(vertex3[0])-1, int(vertex3[2])-1]])
        vertexs.append(currVertexs)
        normals.append(currNormals)

    else:

        for iter in range(numPoses):
            currVertexs = []
            currNormals = []
            with open(myFile+("{:02d}".format(iter+1))+".obj", 'r') as modelFile:       
                for line in modelFile:
                    lineElements = line.split()
                    if len(lineElements) > 0:
                        if lineElements[0] == "v":
                            currVertexs.append([float(lineElements[1]), float(lineElements[2]), invert*float(lineElements[3])])
                        elif lineElements[0] == "vn":
                            auxNormal = [float(lineElements[1]), float(lineElements[2]), invert*float(lineElements[3])]
                            normalLength = math.sqrt(auxNormal[0]*auxNormal[0] + auxNormal[1]*auxNormal[1] + auxNormal[2]*auxNormal[2])
                            if normalLength > 0.0:
                                auxNormal[0] = auxNormal[0]/normalLength
                                auxNormal[1] = auxNormal[1]/normalLength
                                auxNormal[2] = auxNormal[2]/normalLength
                            currNormals.append([auxNormal[0], auxNormal[1], auxNormal[2]])
                        elif lineElements[0] == "f":
                            if iter == 0:
                                vertex1 = lineElements[1].split('/')
                                vertex2 = lineElements[2].split('/')
                                vertex3 = lineElements[3].split('/')
                                faces.append([[int(vertex1[0])-1, int(vertex1[2])-1], [int(vertex2[0])-1, int(vertex2[2])-1], [int(vertex3[0])-1, int(vertex3[2])-1]]) 
            vertexs.append(currVertexs)
            normals.append(currNormals)
    
    auxVertexs = np.array(vertexs)
    coordMax = np.amax(auxVertexs, axis=(0,1))
    coordMin = np.amin(auxVertexs, axis=(0,1))
    center = (coordMax + coordMin)*0.5
    sizeAABB = coordMax - coordMin
    maxSize = np.amax(sizeAABB)*scale

    auxVertexs = (auxVertexs-center)/maxSize
    coordMax = (coordMax-center)/maxSize
    coordMin = (coordMin-center)/maxSize

    return auxVertexs.tolist(), normals, faces, coordMin, coordMax


def generate_rendering_buffers(vertexs, normals, faces):
   
    rendVerts = [[] for i in range(len(vertexs))]
    renderIndexs = []

    indexDict = {}
    for it, face in enumerate(faces):
        vert1 = face[0]
        indexVert1 = len(rendVerts[0])//6
        vert1Str = str(vert1)
        if vert1Str in indexDict:
            indexVert1 = indexDict[vert1Str]
        else:
            indexDict[vert1Str] = indexVert1
            for iter in range(len(vertexs)):
                rendVerts[iter].append(vertexs[iter][vert1[0]][0])
                rendVerts[iter].append(vertexs[iter][vert1[0]][1])
                rendVerts[iter].append(vertexs[iter][vert1[0]][2])
                rendVerts[iter].append(normals[iter][vert1[1]][0])
                rendVerts[iter].append(normals[iter][vert1[1]][1])
                rendVerts[iter].append(normals[iter][vert1[1]][2])

        vert2 = face[1]
        indexVert2 = len(rendVerts[0])//6
        vert2Str = str(vert2)
        if vert2Str in indexDict:
            indexVert2 = indexDict[vert2Str]
        else:
            indexDict[vert2Str] = indexVert2
            for iter in range(len(vertexs)):
                rendVerts[iter].append(vertexs[iter][vert2[0]][0])
                rendVerts[iter].append(vertexs[iter][vert2[0]][1])
                rendVerts[iter].append(vertexs[iter][vert2[0]][2])
                rendVerts[iter].append(normals[iter][vert2[1]][0])
                rendVerts[iter].append(normals[iter][vert2[1]][1])
                rendVerts[iter].append(normals[iter][vert2[1]][2])

        vert3 = face[2]
        indexVert3 = len(rendVerts[0])//6
        vert3Str = str(vert3)
        if vert3Str in indexDict:
            indexVert3 = indexDict[vert3Str]
        else:
            indexDict[vert3Str] = indexVert3
            for iter in range(len(vertexs)):
                rendVerts[iter].append(vertexs[iter][vert3[0]][0])
                rendVerts[iter].append(vertexs[iter][vert3[0]][1])
                rendVerts[iter].append(vertexs[iter][vert3[0]][2])
                rendVerts[iter].append(normals[iter][vert3[1]][0])
                rendVerts[iter].append(normals[iter][vert3[1]][1])
                rendVerts[iter].append(normals[iter][vert3[1]][2])

        renderIndexs.append(indexVert1)
        renderIndexs.append(indexVert2)
        renderIndexs.append(indexVert3)

    return rendVerts, renderIndexs


def compute_triangle_area(position1, position2, position3):
    diffPoint1 = [position1[0] - position2[0], position1[1] - position2[1], position1[2] - position2[2]]
    diffPoint2 = [position1[0] - position3[0], position1[1] - position3[1], position1[2] - position3[2]]
    squared1 = diffPoint1[1]*diffPoint2[2] - diffPoint1[2]*diffPoint2[1]
    squared2 = diffPoint1[2]*diffPoint2[0] - diffPoint1[0]*diffPoint2[2]
    squared3 = diffPoint1[0]*diffPoint2[1] - diffPoint1[1]*diffPoint2[0]
    return 0.5*math.sqrt(squared1*squared1 + squared2*squared2 + squared3*squared3)


def compute_total_area(vertexs, faces):
    totalArea = 0.0
    trianglesAndAreas = []
    for face in faces:
        triangleArea = compute_triangle_area(vertexs[face[0][0]], vertexs[face[1][0]], vertexs[face[2][0]])
        totalArea = totalArea + triangleArea
        trianglesAndAreas.append([face, triangleArea])
    return totalArea, trianglesAndAreas


def get_rnd_var_coors():
    varCoord1 = np.random.random()
    varCoord2 = np.random.random()
    varCoord3 = np.random.random()
    totalSum = varCoord1 + varCoord2 + varCoord3
    varCoord1 = varCoord1/totalSum
    varCoord2 = varCoord2/totalSum
    varCoord3 = varCoord3/totalSum
    
    return [varCoord1, varCoord2, varCoord3]

    
def sample_triangle(varCoord1, varCoord2, varCoord3, vertex1, vertex2, vertex3, normal1, normal2, normal3):
    sampledPoint = [vertex1[0]*varCoord1 + vertex2[0]*varCoord2 + vertex3[0]*varCoord3,
                    vertex1[1]*varCoord1 + vertex2[1]*varCoord2 + vertex3[1]*varCoord3,
                    vertex1[2]*varCoord1 + vertex2[2]*varCoord2 + vertex3[2]*varCoord3]
    sampledNormal = [normal1[0]*varCoord1 + normal2[0]*varCoord2 + normal3[0]*varCoord3,
                    normal1[1]*varCoord1 + normal2[1]*varCoord2 + normal3[1]*varCoord3,
                    normal1[2]*varCoord1 + normal2[2]*varCoord2 + normal3[2]*varCoord3]
    normalLength = math.sqrt(sampledNormal[0]*sampledNormal[0] + sampledNormal[1]*sampledNormal[1] + sampledNormal[2]*sampledNormal[2])
    if normalLength > 0.0:
        sampledNormal[0] = sampledNormal[0]/normalLength
        sampledNormal[1] = sampledNormal[1]/normalLength
        sampledNormal[2] = sampledNormal[2]/normalLength

    return sampledPoint, sampledNormal
    

def sample_mesh(vertexs, normals, faces, numMaxPoints):
    _, trianglesAndAreas = compute_total_area(vertexs[0], faces)
    
    outPoints = []
    outNormals = [] 
        
    samplingCoords = []
    samplingTrians = []
    while len(samplingTrians) < numMaxPoints:
        for triangle in trianglesAndAreas:
            samplingCoords.append(get_rnd_var_coors())
            samplingTrians.append(triangle[0])
        
    for iter in range(len(vertexs)):
        currPoints = []
        currNormals = []
        for varCoords, face in zip(samplingCoords, samplingTrians):
            sampledPoint, sampledNormal = sample_triangle(
                            varCoords[0], varCoords[1], varCoords[2],
                            vertexs[iter][face[0][0]], vertexs[iter][face[1][0]], vertexs[iter][face[2][0]],
                            normals[iter][face[0][1]], normals[iter][face[1][1]], normals[iter][face[2][1]])
            currPoints.append(sampledPoint)
            currNormals.append(sampledNormal)
        
        outPoints.append(currPoints)
        outNormals.append(currNormals)
    
    return np.array(outPoints), np.array(outNormals)
