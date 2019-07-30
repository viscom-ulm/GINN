/////////////////////////////////////////////////////////////////////////////
/// \file renderFrag.glsl
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#version 430

uniform sampler2D posTex;
uniform sampler2D normalTex;
uniform sampler2D matTex;
uniform sampler2D lightTex;
uniform sampler2D colorTex;

uniform float directLight;
uniform vec2 gi;

out vec4 outputColor;

#define PI 3.14159265359

void main()
{
    vec4 positionPix = texelFetch(posTex, ivec2(floor(gl_FragCoord.xy)), 0);
    
    if(positionPix.x < 100.0){
        vec4 colorPix = texelFetch(colorTex, ivec2(floor(gl_FragCoord.xy)), 0);
        vec4 normalPix = texelFetch(normalTex, ivec2(floor(gl_FragCoord.xy)), 0);
        vec4 mat = texelFetch(matTex, ivec2(floor(gl_FragCoord.xy)), 0);
        vec4 light = texelFetch(lightTex, ivec2(floor(gl_FragCoord.xy)), 0);

        vec4 aoOutputColor = vec4(vec3(mat.xyz*(light.xyz+0.5)*colorPix.xyz*(1.0 - directLight)) + vec3(colorPix.xyz*directLight), 1.0);
        vec4 giOutputColor = vec4(vec3(((mat.xyz*light.xyz + mat.xyz*colorPix.xyz*0.5))*(1.0 - directLight)) + vec3((colorPix.xyz*directLight)), 1.0);
        vec4 sssOutputColor = vec4(colorPix.xyz, 1.0)*(1.0 - directLight) + vec4(light.xyz, 1.0)*directLight;
        
        outputColor = aoOutputColor*(1.0 - gi.x) + giOutputColor*(1.0-gi.y)*gi.x + sssOutputColor*gi.y*gi.x;

        if(gi.y < 1.0){
            if (outputColor.x <= 0.0031308) outputColor.x = 12.92 * outputColor.x;
            else outputColor.x = 1.055 * pow(outputColor.x, (1.0/2.4)) - 0.055;

            if (outputColor.y <= 0.0031308) outputColor.y = 12.92 * outputColor.y;
            else outputColor.y = 1.055 * pow(outputColor.y, (1.0/2.4)) - 0.055;

            if (outputColor.z <= 0.0031308) outputColor.z = 12.92 * outputColor.z;
            else outputColor.z = 1.055 * pow(outputColor.z, (1.0/2.4)) - 0.055;
        }
       
    }else{
        outputColor = vec4(0.0,0.0,0.0,1.0);
    }
}