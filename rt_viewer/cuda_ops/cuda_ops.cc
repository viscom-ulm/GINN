/////////////////////////////////////////////////////////////////////////////
/// \file cuda_ops.cc
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <boost/python.hpp>
#include <GL/gl.h>

void computeAOCU(int numCells);
void initInteroperabilityGLCUDACU(
    int width, int height, GLuint inTexIdPts, GLuint inTexIdNormals,
    GLuint outTexId, GLuint inPtsBuffId, GLuint inFeaturesBuffId,
    GLuint inAABBBuff, GLuint inCellIndexsBuff, float radius);
void initInteroperabilityGLCUDAGICU();
void initInteroperabilityGLCUDASSSCU(
    GLuint textMaterial, GLuint textDirectLight,
    float sssParam1, float sssParam2, 
    float sssParam3, float sssParam4);
void initInteroperabilityGLCUDAConvWeightsCU(
    int numfeatures,
    GLuint pWeightsConv1,
    GLuint pWeightsConv2,
    GLuint pWeightsConv3,
    GLuint pBiasesConv1,
    GLuint pBiasesConv2,
    GLuint pBiasesConv3);
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
    GLuint biasesMLP2);


void computeAOTexture(int numCells)
{
   computeAOCU(numCells);
}

void initInteroperabilityGLCUDA(
    int width, int height, int inTexIdPts, int inTexIdNormals,
    int outTexId, int inPtsBuffId, int inFeaturesBuffId,
    int inAABBBuff, int inCellIndexsBuff, float radius)
{
    initInteroperabilityGLCUDACU(width, height, inTexIdPts, 
        inTexIdNormals, outTexId, inPtsBuffId, inFeaturesBuffId,
        inAABBBuff, inCellIndexsBuff, radius);
}

void initInteroperabilityGLCUDAGI()
{
    initInteroperabilityGLCUDAGICU();
}

void initInteroperabilityGLCUDASSS(
    int textMaterial, int textDirectLight,
    float sssParam1, float sssParam2, 
    float sssParam3, float sssParam4)
{
    initInteroperabilityGLCUDASSSCU(
        textMaterial, textDirectLight, 
        sssParam1, sssParam2, sssParam3, sssParam4);
}

void initInteroperabilityGLCUDAConvWeights(
    int numfeatures,
    int pWeightsConv1,
    int pWeightsConv2,
    int pWeightsConv3,
    int pBiasesConv1,
    int pBiasesConv2,
    int pBiasesConv3)
{
    initInteroperabilityGLCUDAConvWeightsCU(numfeatures, pWeightsConv1,
        pWeightsConv2, pWeightsConv3, pBiasesConv1, pBiasesConv2, pBiasesConv3);
}

void initInteroperabilityGLCUDAMLPWeights(
    int meanBN1,
    int varianceBN1,
    int gammaBN1,
    int betaBN1,
    int weightsMLP1,
    int biasesMLP1,
    int meanBN2,
    int varianceBN2,
    int gammaBN2,
    int betaBN2,
    int weightsMLP2,
    int biasesMLP2)
{
    initInteroperabilityGLCUDAMLPWeightsCU(meanBN1, varianceBN1, gammaBN1,
        betaBN1, weightsMLP1, biasesMLP1, meanBN2, varianceBN2, gammaBN2,
        betaBN2, weightsMLP2, biasesMLP2);
}


BOOST_PYTHON_MODULE(screenproc)
{
    using namespace boost::python;
    def("computeAOTexture", computeAOTexture);
    def("initInteroperabilityGLCUDA", initInteroperabilityGLCUDA);
    def("initInteroperabilityGLCUDAGI", initInteroperabilityGLCUDAGI);
    def("initInteroperabilityGLCUDASSS", initInteroperabilityGLCUDASSS);
    def("initInteroperabilityGLCUDAConvWeights", initInteroperabilityGLCUDAConvWeights);
    def("initInteroperabilityGLCUDAMLPWeights", initInteroperabilityGLCUDAMLPWeights);
}