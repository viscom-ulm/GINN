/////////////////////////////////////////////////////////////////////////////
/// \file renderVert.glsl
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#version 430

layout(location = 0)in vec3 position;

void main()
{
    gl_Position = vec4(position, 1.0);
}