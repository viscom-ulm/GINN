/////////////////////////////////////////////////////////////////////////////
/// \file meshFrag.glsl
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#version 430

uniform vec3 objColor;

uniform sampler2DShadow shadowMapTex;

uniform vec3 camPos;
uniform int numVLP;
uniform int numVLPAxis;

in vec3 sPos;
in vec3 sNormal;

out vec4 outputPos;
out vec4 outputNormal;
out vec4 outputMat;
out vec4 outputLight;

#define PI 3.14159265359

layout(std140, binding = 0) buffer VlpPos {vec4 vlpPos[ ];};
layout(std140, binding = 1) buffer VlpIntensities {vec4 vlpIntensities[ ];};
layout(std140, binding = 2) buffer VlpViewMat {mat4 vlpViewMat[ ];};
layout(std140, binding = 3) buffer VlpProjMat {mat4 vlpProjMat[ ];};

void main()
{
    vec3 normNormal = normalize(sNormal);
    vec3 viewVector = normalize(camPos-sPos.xyz);
    vec4 lighting = vec4(0.0);
    for(int vlpIter = 0; vlpIter < numVLP; vlpIter++)
    {
        vec3 curVLPPos = normalize(vlpPos[vlpIter].xyz);
        mat4 curVLPViewMat = transpose(vlpViewMat[vlpIter]);
        mat4 curVLPProjMat = transpose(vlpProjMat[vlpIter]);

        vec4 viewPos = curVLPViewMat*vec4(sPos, 1.0);
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
        lighting += visibility*vlpIntensity*pow(max(dot(normNormal, normalize(curVLPPos+viewVector)), 0.0), 128.0);
    }
    
    outputPos = vec4(sPos, 1.0);
    outputMat = vec4(objColor, 0.0);
    outputNormal = vec4(sNormal, 0.0);
    outputLight = vec4(lighting.xyz/float(numVLP), 1.0);
}