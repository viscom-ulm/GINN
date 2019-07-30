/////////////////////////////////////////////////////////////////////////////
/// \file renderVertShadows.glsl
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#version 430

layout(location = 0)in vec3 position;
layout(location = 1)in vec3 normals;

uniform mat4 viewMatrix;
uniform mat4 projMatrix;

void main()
{
    vec4 viewPos = viewMatrix * vec4(position, 1.0);
    vec4 projPos = projMatrix * viewPos;
    gl_Position = projPos;
}