/////////////////////////////////////////////////////////////////////////////
/// \file directLightCompute.glsl
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#version 430 compatibility

uniform sampler2DShadow shadowMapTex;
uniform int numVLP;
uniform int numVLPAxis;

layout(std140, binding = 0) buffer ptPos {vec4 pointPos[ ];};
layout(std140, binding = 1) buffer ptNormal {vec4 pointNormal[ ];};
layout(std140, binding = 2) buffer outDirectLight {vec4 pointDirectLight[ ];};

layout(std140, binding = 3) buffer VlpPos {vec4 vlpPos[ ];};
layout(std140, binding = 4) buffer VlpIntensities {vec4 vlpIntensities[ ];};
layout(std140, binding = 5) buffer VlpViewMat {mat4 vlpViewMat[ ];};
layout(std140, binding = 6) buffer VlpProjMat {mat4 vlpProjMat[ ];};

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

#define PI 3.14159265359

void main()
{
    uint pointId = gl_GlobalInvocationID.x;
	if(pointId < pointPos.length()){

        vec4 ptPos = pointPos[pointId];
        vec3 normNormal = normalize(pointNormal[pointId].xyz);
        vec4 lighting = vec4(0.0);
        for(int vlpIter = 0; vlpIter < numVLP; vlpIter++)
        {
            vec3 curVLPPos = normalize(vlpPos[vlpIter].xyz);
            mat4 curVLPViewMat = transpose(vlpViewMat[vlpIter]);
            mat4 curVLPProjMat = transpose(vlpProjMat[vlpIter]);

            vec4 viewPos = curVLPViewMat*ptPos;
            vec4 projPos = curVLPProjMat*viewPos;
            vec3 normProjPos = projPos.xyz/projPos.w;

            float lightDot = dot(normNormal, curVLPPos);
            lightDot = max(lightDot, 0.0);
            float bias = -2.0*clamp(0.0015*tan(acos(lightDot)), 0.0, 0.01);
            normProjPos.z += bias;

            vec2 atlasOffset = vec2(floor(vlpIter/numVLPAxis), floor(vlpIter%numVLPAxis));
            normProjPos.xy = normProjPos.xy/float(numVLPAxis) + atlasOffset/float(numVLPAxis);

            vec4 vlpIntensity = vlpIntensities[vlpIter];
            float visibility = texture(shadowMapTex, normProjPos);
            lighting += (visibility*vlpIntensity*lightDot)/vlpIntensity.w;
        }
        pointDirectLight[pointId] = vec4(lighting.xyz/float(numVLP), 1.0);
    }
}