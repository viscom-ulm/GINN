/////////////////////////////////////////////////////////////////////////////
/// \file cuda_ops.cu
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <cstdio>

#define TEX_PTS_PROC_SIZE 16
#define POINT_BLOCK_SIZE 128
#define POINT_BLOCK_PACK_SIZE 256
#define NEIGHBOR_BLOCK_PDF_SIZE 256
#define EXECUTION_BLOCK_MLP_SIZE 128
#define BLOCK_MLP_SIZE 4

//#define PRINT_NUM_PTS

////////////////////////////////////////////////////////////////////////////////// GPU

texture<float4, cudaTextureType2D, cudaReadModeElementType> inTexPts;
texture<float4, cudaTextureType2D, cudaReadModeElementType> inTexNormals;
texture<float4, cudaTextureType2D, cudaReadModeElementType> inTexMaterial;
texture<float4, cudaTextureType2D, cudaReadModeElementType> inTexDirectLight;
surface<void, cudaSurfaceType2D> inOutTexAO;

__constant__ int cellOffsets[27][3];

__global__ void countNeighbors(
    const int width,
    const int height,
    const int pNumCells,
    const float pRadius,
    const float* __restrict__ pAABB, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pCellIndexs,
    int* __restrict__ pOutNeigbors,
    int* __restrict__ pOutNumNeigbors
#ifdef PRINT_NUM_PTS    
    ,
    int* __restrict__ pOutNumPixels) 
#else
    )
#endif
{
    __shared__ int blockTotalNeighbors;

    if(threadIdx.x == 0 && threadIdx.y == 0){
        blockTotalNeighbors = 0;
    }

    __syncthreads();

    int currentIndexX = blockIdx.x*blockDim.x + threadIdx.x;
    int currentIndexY = blockIdx.y*blockDim.y + threadIdx.y;
    if(currentIndexX < width && currentIndexY < height){
        
        float maxAabbSize = max(max(pAABB[3] - pAABB[0], pAABB[4] - pAABB[1]), pAABB[5] - pAABB[2]);
        float cellSize = maxAabbSize/(float)pNumCells;
        float scaledRadius = pRadius;
        
        float4 ptCoordsTex = tex2D(inTexPts, currentIndexX, currentIndexY);

        if(ptCoordsTex.x < 100.0){

            float centralCoords[3] = {ptCoordsTex.x, ptCoordsTex.y, ptCoordsTex.z};
            int xCell = max(min((int)floor((centralCoords[0] - pAABB[0])/cellSize), pNumCells -1), 0);
            int yCell = max(min((int)floor((centralCoords[1] - pAABB[1])/cellSize), pNumCells -1), 0);
            int zCell = max(min((int)floor((centralCoords[2] - pAABB[2])/cellSize), pNumCells -1), 0);

            int neighborIter = 0;
            for(int i = 0; i < 27; ++i)
            {
                int currCellIndex[3] = {xCell+cellOffsets[i][0], yCell+cellOffsets[i][1], zCell+cellOffsets[i][2]};
                if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumCells &&
                    currCellIndex[1] >= 0 && currCellIndex[1] < pNumCells &&
                    currCellIndex[2] >= 0 && currCellIndex[2] < pNumCells)
                {
                    int cellIndexFlat = currCellIndex[0]*pNumCells*pNumCells + currCellIndex[1]*pNumCells + currCellIndex[2];
                    int initIndex = pCellIndexs[cellIndexFlat*2];
                    int endIndex = pCellIndexs[cellIndexFlat*2 + 1];
                    
                    for(int j = initIndex; j < endIndex; ++j)
                    {
                        int currPointIndex = j * 3;
                        float currentCoords[3] = {pPoints[currPointIndex], pPoints[currPointIndex+1], pPoints[currPointIndex+2]};
                        float diffVector[3] = {currentCoords[0] - centralCoords[0], currentCoords[1] - centralCoords[1], currentCoords[2] - centralCoords[2]};
                        float pointDist = sqrt(diffVector[0]*diffVector[0] + diffVector[1]*diffVector[1] + diffVector[2]*diffVector[2]);
                        if(pointDist < scaledRadius){
                            neighborIter++;
                        }
                    }
                }
            }

            int currentIndex = currentIndexX + currentIndexY*width;
            pOutNeigbors[currentIndex] = neighborIter;
            atomicAdd(&blockTotalNeighbors, neighborIter);
#ifdef PRINT_NUM_PTS 
            atomicAdd(pOutNumPixels, 1);
#endif
        }
    }

    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0){
        atomicAdd(&pOutNumNeigbors[0], blockTotalNeighbors);
    }
}


__global__ void computeOffsets(
    const bool pStep1,
    const int pNumOffsets,
    const int pNumOffsets2,
    int* __restrict__ pOutNeighborsOffsets,
    int* __restrict__ pOutNeighborsOffsets2) 
{
    __shared__ int groupOffsets[POINT_BLOCK_PACK_SIZE];

	//Get the local and global counter.
	int currCounter = threadIdx.x;
	int currGlobalCounter = threadIdx.x + blockIdx.x * blockDim.x;

	//Update the shared memory.
	if(currGlobalCounter < pNumOffsets)
		groupOffsets[currCounter] = pOutNeighborsOffsets[currGlobalCounter];
	else
		groupOffsets[currCounter] = 0;

	//SIMD scan.
	for(int i = 1; i <= POINT_BLOCK_PACK_SIZE/2; i*=2)
	{
		__syncthreads();

		//Get the values of the pass.
		int currIndex = currCounter + i;
		int value1 = 0;
		int value2 = 0;
		if(currIndex < POINT_BLOCK_PACK_SIZE){
			value1 = groupOffsets[currCounter];
			value2 = groupOffsets[currIndex];
		}

		__syncthreads();

		//Update with the new value.
		if(currIndex < POINT_BLOCK_PACK_SIZE)
			groupOffsets[currIndex] = value1 + value2;
	}

	__syncthreads();

	//Save the counter into global memory.
	if(currGlobalCounter < pNumOffsets){
		if(currCounter > 0)
			pOutNeighborsOffsets[currGlobalCounter] = groupOffsets[currCounter-1];
		else
			pOutNeighborsOffsets[currGlobalCounter] = 0;
	}

    if(pStep1){
        //Update the offsets buffer.
        if(currCounter == (POINT_BLOCK_PACK_SIZE-1) && blockIdx.x < pNumOffsets2)
            pOutNeighborsOffsets2[blockIdx.x] = groupOffsets[POINT_BLOCK_PACK_SIZE-1];
    }else{
        //Update the second level offset buffer.
        if(currCounter > blockIdx.x && currCounter < pNumOffsets2){
            atomicAdd(&pOutNeighborsOffsets2[currCounter], groupOffsets[POINT_BLOCK_PACK_SIZE-1]);
        }
    }
}

__global__ void findNeighbors(
    const int width,
    const int height,
    const int pNumCells,
    const int pNumNeighs,
    const float pRadius,
    const float* __restrict__ pAABB, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pCellIndexs,
    const int* __restrict__ pStartIndexsOffset,
    const int* __restrict__ pStartIndexsOffset2,
    int* __restrict__ pStartIndexs,
    int* __restrict__ pOutNeigborsPacked) 
{
    int currentIndexX = blockIdx.x*blockDim.x + threadIdx.x;
    int currentIndexY = blockIdx.y*blockDim.y + threadIdx.y;
    if(currentIndexX < width && currentIndexY < height){

        float4 ptCoordsTex = tex2D(inTexPts, currentIndexX, currentIndexY);
        int currentIndex = currentIndexX + currentIndexY*width;
        int offsetIndex = currentIndex/POINT_BLOCK_PACK_SIZE;
        int globalOffsetIndex = offsetIndex/POINT_BLOCK_PACK_SIZE;
        int neighborIndex = pStartIndexs[currentIndex]+pStartIndexsOffset[offsetIndex]+pStartIndexsOffset2[globalOffsetIndex];
        pStartIndexs[currentIndex] = neighborIndex;
        
        if(ptCoordsTex.x < 100.0){

            float maxAabbSize = max(max(pAABB[3] - pAABB[0], pAABB[4] - pAABB[1]), pAABB[5] - pAABB[2]);
            float cellSize = maxAabbSize/(float)pNumCells;
            float scaledRadius = pRadius;

            float centralCoords[3] = {ptCoordsTex.x, ptCoordsTex.y, ptCoordsTex.z};
            int xCell = max(min((int)floor((centralCoords[0] - pAABB[0])/cellSize), pNumCells -1), 0);
            int yCell = max(min((int)floor((centralCoords[1] - pAABB[1])/cellSize), pNumCells -1), 0);
            int zCell = max(min((int)floor((centralCoords[2] - pAABB[2])/cellSize), pNumCells -1), 0);

            int neighborIter = 0;
            for(int i = 0; i < 27; ++i)
            {
                int currCellIndex[3] = {xCell+cellOffsets[i][0], yCell+cellOffsets[i][1], zCell+cellOffsets[i][2]};
                if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumCells &&
                    currCellIndex[1] >= 0 && currCellIndex[1] < pNumCells &&
                    currCellIndex[2] >= 0 && currCellIndex[2] < pNumCells)
                {
                    int cellIndexFlat = currCellIndex[0]*pNumCells*pNumCells + currCellIndex[1]*pNumCells + currCellIndex[2];
                    int initIndex = pCellIndexs[cellIndexFlat*2];
                    int endIndex = pCellIndexs[cellIndexFlat*2 + 1];
                    for(int j = initIndex; j < endIndex; ++j)
                    {
                        int currPointIndex = j * 3;
                        float currentCoords[3] = {pPoints[currPointIndex], pPoints[currPointIndex+1], pPoints[currPointIndex+2]};
                        float diffVector[3] = {currentCoords[0] - centralCoords[0], currentCoords[1] - centralCoords[1], currentCoords[2] - centralCoords[2]};
                        float pointDist = sqrt(diffVector[0]*diffVector[0] + diffVector[1]*diffVector[1] + diffVector[2]*diffVector[2]);
                        if(pointDist < scaledRadius){
                            pOutNeigborsPacked[neighborIndex*2 + neighborIter] = j;
                            pOutNeigborsPacked[neighborIndex*2 + neighborIter + 1] = currentIndex;
                            neighborIter+=2;
                        }
                    }
                }
            }
        }
    }
}

__global__ void computePDFs(
    const float pWindow,
    const int numSamples,
    const int pNumNeighbors,
    const float pRadius,
    const float* __restrict__ pAABB,
    const float* __restrict__ pPoints, 
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    float* __restrict__ pOutPDFs) 
{
    int currentNeighborIndex = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPoint = pNeigbors[neighborIndex]*3;
        float currPointCoords[3] = {pPoints[currentPoint], pPoints[currentPoint+1], pPoints[currentPoint+2]};

        float scaledRadius = pRadius;
        
        int centralPoint = pNeigbors[neighborIndex+1];
        int initIter = pStartIndexs[centralPoint];
        int endIter = (centralPoint < numSamples-1)?pStartIndexs[centralPoint+1]:pNumNeighbors;

        const float h = pWindow;
        const float invH = 1/h;
        const float invRadH = 1.0/(scaledRadius*h);
        float currPdf = 0.0;
        int iter = initIter;
        while(iter < endIter)
        {
            int iterPoint = pNeigbors[iter*2]*3;
            float iterPointCoords[3] = {pPoints[iterPoint], pPoints[iterPoint+1], pPoints[iterPoint+2]};
            float diff [3] = {
                (iterPointCoords[0] - currPointCoords[0])*invRadH, 
                (iterPointCoords[1] - currPointCoords[1])*invRadH, 
                (iterPointCoords[2] - currPointCoords[2])*invRadH};
            float gaussVal = invH*((0.39894228)*exp((-0.5)*diff[0]*diff[0]));
            gaussVal = gaussVal*invH*((0.39894228)*exp((-0.5)*diff[1]*diff[1]));
            gaussVal = gaussVal*invH*((0.39894228)*exp((-0.5)*diff[2]*diff[2]));
            currPdf += gaussVal;
            iter++;
        }
        
        pOutPDFs[currentNeighborIndex] = (currPdf)/((float)endIter-initIter);
    }
}

__device__ void evaluateMLPNoComb(
    const int pThreadId,
    const int pOffset,
    const int pTotalBlocks,
    const int pNumFeatures,
    const int pFeatureIndex,
    const int pOutFeatureIndex,
    const float pNumSamples,
    const float pCurrentPDF,
    const float pCurrPointCoords[3],
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pFeatures,
    float* __restrict__ pTmpVector1,
    float* __restrict__ pTmpVector2,
    float* __restrict__ pOutFeatures)
{
    pTmpVector1[pThreadId] = max(pCurrPointCoords[0]*pWeightsHidd1[pThreadId*3] + 
                        pCurrPointCoords[1]*pWeightsHidd1[pThreadId*3 + 1] +
                        pCurrPointCoords[2]*pWeightsHidd1[pThreadId*3 + 2] +
                        pBiasHidd1[pThreadId], 0.0);

    __syncthreads();

    //Compute output second layer.
    float auxResult = 0.0;
    for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
    {
        auxResult += pTmpVector1[j]*pWeightsHidd2[pThreadId*BLOCK_MLP_SIZE + j];
    }
    pTmpVector2[pThreadId] = max(auxResult + pBiasHidd2[pThreadId], 0.0);

    __syncthreads();
    
    //Compute output layer.
    if((pOffset+pThreadId) < pNumFeatures){
        auxResult = 0.0;
        for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
        {
            auxResult += pTmpVector2[j]*pWeightsOut[pThreadId*BLOCK_MLP_SIZE + j];  
        }
        auxResult = auxResult + pBiasOut[pThreadId];
        int currFeatureIndex = (pOffset+pThreadId)%pNumFeatures;
        atomicAdd(&pOutFeatures[pOutFeatureIndex+currFeatureIndex], 
            (pFeatures[pFeatureIndex+currFeatureIndex]*auxResult)/(pCurrentPDF*pNumSamples));
    }
}


__global__ void evaluateMLPNoCombinKernel(
    const int pNumChannels,
    const int pWidth,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumFeatures,
    const float pRadius,
    const float* __restrict__ pAABB,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pPoints, 
    const float* __restrict__ pFeatures,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pOutFeatures) 
{
    extern __shared__ float mlpIntermediateRes[];

    int neuronsOut = pNumFeatures;
    int numBlocksXNeigh = neuronsOut/BLOCK_MLP_SIZE;
    numBlocksXNeigh += (neuronsOut%BLOCK_MLP_SIZE != 0)?1:0;

    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currChannel = currentIndex%pNumChannels;
    currentIndex = currentIndex/pNumChannels;
    int currentNeighborIndex = currentIndex/(numBlocksXNeigh*BLOCK_MLP_SIZE);
    int offset = currentIndex%(numBlocksXNeigh*BLOCK_MLP_SIZE);
    offset = offset - offset%BLOCK_MLP_SIZE;
    int threadId = threadIdx.x%BLOCK_MLP_SIZE;
    int threadOffset = threadIdx.x - threadId;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];
        int pixelCoordX = centralPointIndex%pWidth;
        int pixelCoordY = centralPointIndex/pWidth;
        float4 ptCoordsTex = tex2D(inTexPts, pixelCoordX, pixelCoordY);

        float scaledRadius = pRadius;
        
        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - ptCoordsTex.x)/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - ptCoordsTex.y)/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - ptCoordsTex.z)/scaledRadius};
        float currPDF = pPDFs[currentNeighborIndex];
        int initIter = pStartIndexs[centralPointIndex];
        int endIter = (centralPointIndex < pNumPoints-1)?pStartIndexs[centralPointIndex+1]:pNumNeighbors;
        float numNeighbors = (float)(endIter-initIter);
        int featureIndex = currentPointIndex*pNumFeatures*pNumChannels + pNumFeatures*currChannel;
        int outFeatureIndex = centralPointIndex*pNumFeatures*pNumChannels + pNumFeatures*currChannel;
        
        float* temporalMemory1 = &mlpIntermediateRes[threadOffset];
        float* temporalMemory2 = &mlpIntermediateRes[EXECUTION_BLOCK_MLP_SIZE + threadOffset];

        evaluateMLPNoComb(threadId, offset, numBlocksXNeigh, pNumFeatures,
            featureIndex, outFeatureIndex, numNeighbors, currPDF, currPointCoords, 
            &pWeightsHidd1[offset*3], &pWeightsHidd2[offset*BLOCK_MLP_SIZE], &pWeightsOut[offset*BLOCK_MLP_SIZE], 
            &pBiasHidd1[offset], &pBiasHidd2[offset], &pBiasOut[offset], 
            pFeatures, temporalMemory1, temporalMemory2, pOutFeatures);
    }
}

__global__ void evaluateFinalMLP(
    const bool pGI,
    const bool pSSS,
    const int pNumChannels,
    const int pWidth,
    const int pHeight,
    const int pNumFeatures,
    const float sssParams1,
    const float sssParams2,
    const float sssParams3,
    const float sssParams4,
    const float* __restrict__ pWeights1,
    const float* __restrict__ pWeights2,
    const float* __restrict__ pBias1,
    const float* __restrict__ pBias2,
    const float* __restrict__ pMeanBN1,
    const float* __restrict__ pMeanBN2,
    const float* __restrict__ pVarianceBN1,
    const float* __restrict__ pVarianceBN2,
    const float* __restrict__ pGammaBN1,
    const float* __restrict__ pGammaBN2,
    const float* __restrict__ pBetaBN1,
    const float* __restrict__ pBetaBN2,
    const float* __restrict__ pAbstractFeatures) 
{
    extern __shared__ float mlpIntermediateRes[];

    int currentIndexX = blockIdx.x*blockDim.x + threadIdx.x;
    int currentIndexY = blockIdx.y*blockDim.y + threadIdx.y;
    if(currentIndexX < pWidth && currentIndexY < pHeight){
        float finalVal[3] = {0.0f, 0.0f, 0.0f};
        float4 ptCoordsTex = tex2D(inTexPts, currentIndexX, currentIndexY);
        if(ptCoordsTex.x < 100.0){
            int currentIndex = currentIndexX + currentIndexY*pWidth;
            int localIndex = threadIdx.x + threadIdx.y*blockDim.x;

            float* tmpPtr1 = &mlpIntermediateRes[localIndex*pNumFeatures];
            float* tmpPtr2 = &mlpIntermediateRes[((blockDim.x*blockDim.y)+localIndex)*pNumFeatures];

            float4 normalTex = tex2D(inTexNormals, currentIndexX, currentIndexY);
            float normlength = sqrt(normalTex.x*normalTex.x+normalTex.y*normalTex.y+normalTex.z*normalTex.z);
            normalTex.x = normalTex.x/normlength;
            normalTex.y = normalTex.y/normlength;
            normalTex.z = normalTex.z/normlength;

            float material[3];
            float dirLight[3];
            if(pSSS){
                float4 materialAux = tex2D(inTexMaterial, currentIndexX, currentIndexY);
                material[0] = materialAux.x;
                material[1] = materialAux.y;
                material[2] = materialAux.z;
                
                float4 dirLightAux = tex2D(inTexDirectLight, currentIndexX, currentIndexY);
                dirLight[0] = dirLightAux.x;
                dirLight[1] = dirLightAux.y;
                dirLight[2] = dirLightAux.z;
            }

            for(int currChannel = 0; currChannel < pNumChannels; ++currChannel)
            {
                for(int i = 0; i < pNumFeatures; i++){
                    float auxVariance = pGammaBN1[i]/sqrt(pVarianceBN1[i]+0.0001);
                    float auxVal = ((pAbstractFeatures[currentIndex*pNumFeatures*pNumChannels + currChannel*pNumFeatures + i]
                                    - pMeanBN1[i])*auxVariance)+pBetaBN1[i];
                    
                    tmpPtr1[i] = max(auxVal, 0.0)-0.2*max(-auxVal, 0.0);
                }

                float sssParams[4] = {sssParams1, sssParams2, sssParams3, sssParams4};
                for(int i = 0; i < pNumFeatures; i++){
                    float auxVal = 0.0;
                    for(int j = 0; j < pNumFeatures; j++){
                        auxVal += tmpPtr1[j]*pWeights1[j*pNumFeatures + i];
                    }
                    auxVal += normalTex.x*pWeights1[(pNumFeatures)*pNumFeatures + i];
                    auxVal += normalTex.y*pWeights1[(pNumFeatures+1)*pNumFeatures + i];
                    auxVal += normalTex.z*pWeights1[(pNumFeatures+2)*pNumFeatures + i];
                    if(pSSS){
                        auxVal += dirLight[currChannel]*pWeights1[(pNumFeatures+3)*pNumFeatures + i];
                        auxVal += material[currChannel]*pWeights1[(pNumFeatures+4)*pNumFeatures + i];
                        auxVal += sssParams[currChannel]*pWeights1[(pNumFeatures+5)*pNumFeatures + i];
                        auxVal += sssParams[3]*pWeights1[(pNumFeatures+6)*pNumFeatures + i];
                    }
                    auxVal += pBias1[i];
                    float auxVariance = pGammaBN2[i]/sqrt(pVarianceBN2[i]+0.0001);
                    auxVal = ((auxVal - pMeanBN2[i])*auxVariance)+pBetaBN2[i];
                    tmpPtr2[i] = max(auxVal, 0.0)-0.2*max(-auxVal, 0.0);
                }

                for(int i = 0; i < pNumFeatures; i++){
                    finalVal[currChannel] += tmpPtr2[i]*pWeights2[i];
                }
                finalVal[currChannel] += pBias2[0];
                finalVal[currChannel] = finalVal[currChannel];
            }
        }

        for(int currChannel = pNumChannels; currChannel < 3; ++currChannel){
            finalVal[currChannel] = finalVal[0];
            finalVal[currChannel] = finalVal[0];
        }

        surf2Dwrite(make_float4(finalVal[0], finalVal[1], finalVal[2], 1.0), inOutTexAO, currentIndexX*sizeof(float)*4, currentIndexY);
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline dim3 computeBlockGrid(const unsigned long long int pNumElements, const int pNumThreads)
{
    dim3 finalDimension(pNumElements/pNumThreads, 1, 1);
    finalDimension.x += (pNumElements%pNumThreads!= 0)?1:0;
    while(finalDimension.x >= 65536){
        finalDimension.y *= 2;
        int auxDim = finalDimension.x/2;
        auxDim += (finalDimension.x%2!=0)?1:0;
        finalDimension.x = auxDim;
    }

    while(finalDimension.y >= 65536){
        finalDimension.z *= 2;
        int auxDim = finalDimension.y/2;
        auxDim += (finalDimension.y%2!=0)?1:0;
        finalDimension.y = auxDim;
    }
    return finalDimension;
}

//Declare the needed variables.
int width_;
int height_;
float radius_;
cudaGraphicsResource* inTexPtsRes_;
cudaGraphicsResource* inTexNormalsRes_;
cudaGraphicsResource* outTexAORes_;
cudaGraphicsResource* inPtsBuffRes_;
cudaGraphicsResource* inFeaturesBuffRes_;
cudaGraphicsResource* inAABBBuffRes_;
cudaGraphicsResource* inCellIndexsBuffRes_;

//Initialize resources.
void initInteroperabilityGLCUDACU(
    int width, int height, GLuint inTexIdPts, GLuint inTexIdNormals,
    GLuint outTexId, GLuint inPtsBuffId, GLuint inFeaturesBuffId,
    GLuint inAABBBuff, GLuint inCellIndexsBuff, float radius)
{
    width_ = width;
    height_ = height;
    radius_ = radius;
    gpuErrchk(cudaGraphicsGLRegisterImage(&inTexPtsRes_, inTexIdPts, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterImage(&inTexNormalsRes_, inTexIdNormals, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterImage(&outTexAORes_, outTexId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&inPtsBuffRes_, inPtsBuffId, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&inFeaturesBuffRes_, inFeaturesBuffId, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&inAABBBuffRes_, inAABBBuff, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&inCellIndexsBuffRes_, inCellIndexsBuff, cudaGraphicsMapFlagsReadOnly));
}

bool gi_ = false;
void initInteroperabilityGLCUDAGICU()
{
    gi_ = true;
    
}

bool sss_ = false;
float sssParams[4];
cudaGraphicsResource* inTexMaterialRes_;
cudaGraphicsResource* inTexDirectLightRes_;
void initInteroperabilityGLCUDASSSCU(
    GLuint textMaterial, GLuint textDirectLight,
    float sssParam1, float sssParam2, 
    float sssParam3, float sssParam4)
{
    sss_ = true;
    sssParams[0] = sssParam1;
    sssParams[1] = sssParam2;
    sssParams[2] = sssParam3;
    sssParams[3] = sssParam4;
    gpuErrchk(cudaGraphicsGLRegisterImage(&inTexMaterialRes_, textMaterial, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    gpuErrchk(cudaGraphicsGLRegisterImage(&inTexDirectLightRes_, textDirectLight, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

int numFeatures_;
cudaGraphicsResource* weightsConv1_;
cudaGraphicsResource* weightsConv2_;
cudaGraphicsResource* weightsConv3_;
cudaGraphicsResource* biasesConv1_;
cudaGraphicsResource* biasesConv2_;
cudaGraphicsResource* biasesConv3_;

void initInteroperabilityGLCUDAConvWeightsCU(
    int numfeatures,
    GLuint pWeightsConv1,
    GLuint pWeightsConv2,
    GLuint pWeightsConv3,
    GLuint pBiasesConv1,
    GLuint pBiasesConv2,
    GLuint pBiasesConv3)
{
    numFeatures_ = numfeatures;
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&weightsConv1_, pWeightsConv1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&weightsConv2_, pWeightsConv2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&weightsConv3_, pWeightsConv3, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&biasesConv1_, pBiasesConv1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&biasesConv2_, pBiasesConv2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&biasesConv3_, pBiasesConv3, cudaGraphicsMapFlagsReadOnly));
}

cudaGraphicsResource* meanBN1_;
cudaGraphicsResource* varianceBN1_;
cudaGraphicsResource* gammaBN1_;
cudaGraphicsResource* betaBN1_;
cudaGraphicsResource* weightsMLP1_;
cudaGraphicsResource* biasesMLP1_;
cudaGraphicsResource* meanBN2_;
cudaGraphicsResource* varianceBN2_;
cudaGraphicsResource* gammaBN2_;
cudaGraphicsResource* betaBN2_;
cudaGraphicsResource* weightsMLP2_;
cudaGraphicsResource* biasesMLP2_;

void initInteroperabilityGLCUDAMLPWeightsCU(
    GLuint meanBN1,
    GLuint varianceBN1,
    GLuint gammaBN1,
    GLuint betaBN1,
    GLuint weightsMLP1,
    GLuint biasesMLP1,
    GLuint meanBN2,
    GLuint varianceBN2,
    GLuint gammaBN2,
    GLuint betaBN2,
    GLuint weightsMLP2,
    GLuint biasesMLP2)
{
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&meanBN1_, meanBN1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&varianceBN1_, varianceBN1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&gammaBN1_, gammaBN1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&betaBN1_, betaBN1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&weightsMLP1_, weightsMLP1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&biasesMLP1_, biasesMLP1, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&meanBN2_, meanBN2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&varianceBN2_, varianceBN2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&gammaBN2_, gammaBN2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&betaBN2_, betaBN2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&weightsMLP2_, weightsMLP2, cudaGraphicsMapFlagsReadOnly));
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&biasesMLP2_, biasesMLP2, cudaGraphicsMapFlagsReadOnly));
}

//Compute AO.
void computeAOCU(int numCells)
{
    //Get the resources.

    //Input textures.
    cudaArray* mTexPts;
    gpuErrchk(cudaGraphicsMapResources(1, &inTexPtsRes_, 0));
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&mTexPts, inTexPtsRes_, 0, 0));
    gpuErrchk(cudaBindTextureToArray(inTexPts, mTexPts));

    cudaArray* mTexNormals;
    gpuErrchk(cudaGraphicsMapResources(1, &inTexNormalsRes_, 0));
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&mTexNormals, inTexNormalsRes_, 0, 0));
    gpuErrchk(cudaBindTextureToArray(inTexNormals, mTexNormals));
    
    if(sss_){
        cudaArray* mTexMat;
        gpuErrchk(cudaGraphicsMapResources(1, &inTexMaterialRes_, 0));
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&mTexMat, inTexMaterialRes_, 0, 0));
        gpuErrchk(cudaBindTextureToArray(inTexMaterial, mTexMat));

        cudaArray* mTexDirectLight;
        gpuErrchk(cudaGraphicsMapResources(1, &inTexDirectLightRes_, 0));
        gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&mTexDirectLight, inTexDirectLightRes_, 0, 0));
        gpuErrchk(cudaBindTextureToArray(inTexDirectLight, mTexDirectLight));
    }
    
    cudaArray* mOutTexAO;
    gpuErrchk(cudaGraphicsMapResources(1, &outTexAORes_, 0));
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&mOutTexAO, outTexAORes_, 0, 0));
    gpuErrchk(cudaBindSurfaceToArray(inOutTexAO, mOutTexAO));

    float* mPtsPtr;
    size_t mPtsPtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &inPtsBuffRes_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mPtsPtr, &mPtsPtrSize, inPtsBuffRes_));

    float* mFeaturesPtr;
    size_t mFeaturesPtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &inFeaturesBuffRes_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mFeaturesPtr, &mFeaturesPtrSize, inFeaturesBuffRes_));

    float* mAABBPtr;
    size_t mAABBPtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &inAABBBuffRes_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mAABBPtr, &mAABBPtrSize, inAABBBuffRes_));

    int* mCellIndexsPtr;
    size_t mCellIndexsPtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &inCellIndexsBuffRes_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mCellIndexsPtr, &mCellIndexsPtrSize, inCellIndexsBuffRes_));

    //Count the neighbors.

    //Init device symbols.
    int cellOffsetsCPU[27][3] = {
        {1, 1, 1},{0, 1, 1},{-1, 1, 1},
        {1, 0, 1},{0, 0, 1},{-1, 0, 1},
        {1, -1, 1},{0, -1, 1},{-1, -1, 1},
        {1, 1, 0},{0, 1, 0},{-1, 1, 0},
        {1, 0, 0},{0, 0, 0},{-1, 0, 0},
        {1, -1, 0},{0, -1, 0},{-1, -1, 0},
        {1, 1, -1},{0, 1, -1},{-1, 1, -1},
        {1, 0, -1},{0, 0, -1},{-1, 0, -1},
        {1, -1, -1},{0, -1, -1},{-1, -1, -1}};
    cudaMemcpyToSymbol(cellOffsets, cellOffsetsCPU, 27*3*sizeof(int));

    int numPts = width_*height_;
    dim3 blockSizeNeighs(TEX_PTS_PROC_SIZE, TEX_PTS_PROC_SIZE, 1);
    dim3 gridSizeNeighs(width_/TEX_PTS_PROC_SIZE, height_/TEX_PTS_PROC_SIZE, 1);
    if(width_%TEX_PTS_PROC_SIZE != 0){
        gridSizeNeighs.x += 1;
    }
    if(height_%TEX_PTS_PROC_SIZE != 0){
        gridSizeNeighs.y += 1;
    }

    //Find the neighbors for each point.
    int* totalNeighbors;
    gpuErrchk(cudaMalloc(&totalNeighbors, sizeof(int)));
    cudaMemset(totalNeighbors, 0, sizeof(int));
#ifdef PRINT_NUM_PTS 
    int* totalPixels;
    gpuErrchk(cudaMalloc(&totalPixels, sizeof(int)));
    cudaMemset(totalPixels, 0, sizeof(int));
#endif
    int* startIndexs;
    gpuErrchk(cudaMalloc(&startIndexs, sizeof(int)*numPts));
    cudaMemset(startIndexs, 0, sizeof(int)*numPts);

    countNeighbors<<<gridSizeNeighs, blockSizeNeighs>>>(width_, height_, numCells, 
        radius_, mAABBPtr, mPtsPtr, mCellIndexsPtr, startIndexs, totalNeighbors
#ifdef PRINT_NUM_PTS 
        , totalPixels);
#else
        );
#endif

    gpuErrchk(cudaPeekAtLastError());

    int totalNeighborsCPU = 0;
    cudaMemcpy(&totalNeighborsCPU, totalNeighbors, sizeof(int), cudaMemcpyDeviceToHost);

#ifdef PRINT_NUM_PTS 
    int totalPixelsCPU = 0;
    cudaMemcpy(&totalPixelsCPU, totalPixels, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, "Num pts: %d | Num neighs: %d\n", totalPixelsCPU, totalNeighborsCPU);
#endif

    gpuErrchk(cudaFree(totalNeighbors));

    int numBlocksPointsPack = numPts/POINT_BLOCK_PACK_SIZE;
    numBlocksPointsPack += (numPts%POINT_BLOCK_PACK_SIZE != 0)?1:0;
    int numBlocksPointsPack2 = numBlocksPointsPack/POINT_BLOCK_PACK_SIZE;
    numBlocksPointsPack2 += (numBlocksPointsPack%POINT_BLOCK_PACK_SIZE != 0)?1:0;

    int* auxBuffOffsets;
    int* auxBuffOffsets2;
    int* packedIndexs;
    gpuErrchk(cudaMalloc(&auxBuffOffsets, sizeof(int)*numBlocksPointsPack));
    gpuErrchk(cudaMalloc(&auxBuffOffsets2, sizeof(int)*numBlocksPointsPack2));
    gpuErrchk(cudaMalloc(&packedIndexs, sizeof(int)*totalNeighborsCPU*2));
    gpuErrchk(cudaMemset(auxBuffOffsets, 0, sizeof(int)*numBlocksPointsPack));
    gpuErrchk(cudaMemset(auxBuffOffsets2, 0, sizeof(int)*numBlocksPointsPack2));
    
    computeOffsets<<<numBlocksPointsPack, POINT_BLOCK_PACK_SIZE>>>(true, numPts, numBlocksPointsPack, startIndexs, auxBuffOffsets);
    
    gpuErrchk(cudaPeekAtLastError());

    computeOffsets<<<numBlocksPointsPack2, POINT_BLOCK_PACK_SIZE>>>(false, numBlocksPointsPack, numBlocksPointsPack2, auxBuffOffsets, auxBuffOffsets2);

    gpuErrchk(cudaPeekAtLastError());

    int numBlocksPoints = numPts/POINT_BLOCK_SIZE;
    numBlocksPoints += (numPts%POINT_BLOCK_SIZE != 0)?1:0;
    findNeighbors<<<gridSizeNeighs, blockSizeNeighs>>>(width_, height_, numCells, totalNeighborsCPU, 
        radius_, mAABBPtr, mPtsPtr, mCellIndexsPtr, auxBuffOffsets, auxBuffOffsets2, startIndexs, packedIndexs);

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaFree(auxBuffOffsets));
    gpuErrchk(cudaFree(auxBuffOffsets2));

    gpuErrchk(cudaPeekAtLastError());

    //Compute the PDFs
    float* pfds;
    gpuErrchk(cudaMalloc(&pfds, sizeof(float)*totalNeighborsCPU));
    dim3 gridDimension = computeBlockGrid(totalNeighborsCPU, NEIGHBOR_BLOCK_PDF_SIZE);

    computePDFs<<<gridDimension, NEIGHBOR_BLOCK_PDF_SIZE>>>(0.2, numPts, totalNeighborsCPU, 
        radius_, mAABBPtr, mPtsPtr, startIndexs, packedIndexs, pfds);

    gpuErrchk(cudaPeekAtLastError());

    //Compute the spatial convolution.
    //Get Resources GL.
    float* mWeightsConv1Ptr;
    size_t mWeightsConv1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &weightsConv1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mWeightsConv1Ptr, &mWeightsConv1PtrSize, weightsConv1_));

    float* mWeightsConv2Ptr;
    size_t mWeightsConv2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &weightsConv2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mWeightsConv2Ptr, &mWeightsConv2PtrSize, weightsConv2_));

    float* mWeightsConv3Ptr;
    size_t mWeightsConv3PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &weightsConv3_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mWeightsConv3Ptr, &mWeightsConv3PtrSize, weightsConv3_));

    float* mBiasesConv1Ptr;
    size_t mBiasesConv1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &biasesConv1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBiasesConv1Ptr, &mBiasesConv1PtrSize, biasesConv1_));

    float* mBiasesConv2Ptr;
    size_t mBiasesConv2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &biasesConv2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBiasesConv2Ptr, &mBiasesConv2PtrSize, biasesConv2_));

    float* mBiasesConv3Ptr;
    size_t mBiasesConv3PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &biasesConv3_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBiasesConv3Ptr, &mBiasesConv3PtrSize, biasesConv3_));

    float* mMeanBN1Ptr;
    size_t mMeanBN1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &meanBN1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mMeanBN1Ptr, &mMeanBN1PtrSize, meanBN1_));

    float* mVarianceBN1Ptr;
    size_t mVarianceBN1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &varianceBN1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mVarianceBN1Ptr, &mVarianceBN1PtrSize, varianceBN1_));

    float* mGammaBN1Ptr;
    size_t mGammaBN1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &gammaBN1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mGammaBN1Ptr, &mGammaBN1PtrSize, gammaBN1_));

    float* mBetaBN1Ptr;
    size_t mBetaBN1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &betaBN1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBetaBN1Ptr, &mBetaBN1PtrSize, betaBN1_));

    float* mWeightsMLP1Ptr;
    size_t mWeightsMLP1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &weightsMLP1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mWeightsMLP1Ptr, &mWeightsMLP1PtrSize, weightsMLP1_));

    float* mBiasesMLP1Ptr;
    size_t mBiasesMLP1PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &biasesMLP1_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBiasesMLP1Ptr, &mBiasesMLP1PtrSize, biasesMLP1_));

    float* mMeanBN2Ptr;
    size_t mMeanBN2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &meanBN2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mMeanBN2Ptr, &mMeanBN2PtrSize, meanBN2_));

    float* mVarianceBN2Ptr;
    size_t mVarianceBN2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &varianceBN2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mVarianceBN2Ptr, &mVarianceBN2PtrSize, varianceBN2_));

    float* mGammaBN2Ptr;
    size_t mGammaBN2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &gammaBN2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mGammaBN2Ptr, &mGammaBN2PtrSize, gammaBN2_));

    float* mBetaBN2Ptr;
    size_t mBetaBN2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &betaBN2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBetaBN2Ptr, &mBetaBN2PtrSize, betaBN2_));

    float* mWeightsMLP2Ptr;
    size_t mWeightsMLP2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &weightsMLP2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mWeightsMLP2Ptr, &mWeightsMLP2PtrSize, weightsMLP2_));

    float* mBiasesMLP2Ptr;
    size_t mBiasesMLP2PtrSize;
    gpuErrchk(cudaGraphicsMapResources(1, &biasesMLP2_, 0));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&mBiasesMLP2Ptr, &mBiasesMLP2PtrSize, biasesMLP2_));

    int numChannels = 1;
    if(gi_){
        numChannels = 3;
    }

    float* abstractFeatures;
    gpuErrchk(cudaMalloc(&abstractFeatures, numPts*numFeatures_*numChannels*sizeof(float)));
    gpuErrchk(cudaMemset(abstractFeatures, 0, numPts*numFeatures_*numChannels*sizeof(float)));

    int numBlocksPerPoint = (numFeatures_)/BLOCK_MLP_SIZE;
    numBlocksPerPoint += ((numFeatures_)%BLOCK_MLP_SIZE != 0)?1:0;
    dim3 gridDimension2 = computeBlockGrid(
        (unsigned long long int)totalNeighborsCPU*
        (unsigned long long int)numBlocksPerPoint*
        (unsigned long long int)BLOCK_MLP_SIZE*
        (unsigned long long int)numChannels, EXECUTION_BLOCK_MLP_SIZE);

    evaluateMLPNoCombinKernel<<<gridDimension2, EXECUTION_BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE*2*sizeof(float)>>>(
        numChannels, width_, numPts, totalNeighborsCPU, numFeatures_, radius_, mAABBPtr, 
        mWeightsConv1Ptr, mWeightsConv2Ptr, mWeightsConv3Ptr, mBiasesConv1Ptr, mBiasesConv2Ptr, mBiasesConv3Ptr, 
        mPtsPtr, mFeaturesPtr, startIndexs, packedIndexs, pfds, abstractFeatures);

    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaFree(pfds));
    gpuErrchk(cudaFree(startIndexs));
    gpuErrchk(cudaFree(packedIndexs));

    int memoryUsed = blockSizeNeighs.x*blockSizeNeighs.y*blockSizeNeighs.z*2*(numFeatures_)*sizeof(float);
    evaluateFinalMLP<<<gridSizeNeighs, blockSizeNeighs, memoryUsed>>>(
        gi_, sss_, numChannels, width_, height_, numFeatures_, sssParams[0], sssParams[1], sssParams[2],
        sssParams[3], mWeightsMLP1Ptr, mWeightsMLP2Ptr, mBiasesMLP1Ptr, mBiasesMLP2Ptr, mMeanBN1Ptr, 
        mMeanBN2Ptr, mVarianceBN1Ptr, mVarianceBN2Ptr, mGammaBN1Ptr, mGammaBN2Ptr, mBetaBN1Ptr, mBetaBN2Ptr, abstractFeatures);


    //Clean and release resources.
    gpuErrchk(cudaFree(abstractFeatures));
    gpuErrchk(cudaGraphicsUnmapResources(1, &weightsConv1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &weightsConv2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &weightsConv3_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &biasesConv1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &biasesConv2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &biasesConv3_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &meanBN1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &varianceBN1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &gammaBN1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &betaBN1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &weightsMLP1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &biasesMLP1_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &meanBN2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &varianceBN2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &gammaBN2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &betaBN2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &weightsMLP2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &biasesMLP2_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inTexPtsRes_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inTexNormalsRes_, 0));
    if(sss_){
        gpuErrchk(cudaGraphicsUnmapResources(1, &inTexMaterialRes_, 0));
        gpuErrchk(cudaGraphicsUnmapResources(1, &inTexDirectLightRes_, 0));
    }
    gpuErrchk(cudaGraphicsUnmapResources(1, &outTexAORes_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inPtsBuffRes_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inFeaturesBuffRes_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inAABBBuffRes_, 0));
    gpuErrchk(cudaGraphicsUnmapResources(1, &inCellIndexsBuffRes_, 0));
}