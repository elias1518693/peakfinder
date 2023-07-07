in vec2 texcoords;
out vec4 fragColor;

uniform sampler2D inputTexture;
uniform vec2 imageSize;
uniform float fov;
vec2 cubeMapTo2D(in vec3 cubeVec) {
    float absX = abs(cubeVec.x);
    float absY = abs(cubeVec.y);
    float absZ = abs(cubeVec.z);

    float x, y;

    if (absY >= absX && absY >= absZ) { // Y is the biggest
        if (cubeVec.y > 0.0) {
            //mitte
            x = cubeVec.x;
            y = -cubeVec.z;
        }
        else {
            //rand
           discard;
        }

    }
    else { // Z is the biggest
        discard;
    }

    // Map [-1, 1] to [0, 1]
    x = (x + 1.0) * 0.5;
    y = (y + 1.0) * 0.5;

    return vec2(x, y);
 }
vec2 calcCoordinates(vec2 uv){
    vec2 thetaphi = texcoords * vec2(3.1415926535897932384626433832795, 1.5707963267948966192313216916398);
    vec3 rayDirection = vec3(cos(thetaphi.y) * cos(thetaphi.x), cos(thetaphi.y) * sin(thetaphi.x), sin(thetaphi.y));
    float pi = 3.1415926535897932384626433f;
    float x = sin(uv.x * pi);
    float y = cos(uv.x * pi);
    float z = uv.y * tan(fov/2);
    return cubeMapTo2D(vec3(x,y,z));

}

void make_kernel(inout float n[9], sampler2D tex, vec2 coord)
{
        vec2 texSize = textureSize(tex, 0);
        float w = 1.0 / texSize.x;
        float h = 1.0 / texSize.y;

        n[0] = dot(texture2D(tex,calcCoordinates( coord + vec2( -w, -h))).xyz, vec3(0.299, 0.587, 0.114));
        n[1] = dot(texture2D(tex,calcCoordinates( coord + vec2(0.0, -h))).xyz, vec3(0.299, 0.587, 0.114));
        n[2] = dot(texture2D(tex,calcCoordinates( coord + vec2(  w, -h))).xyz, vec3(0.299, 0.587, 0.114));
        n[3] = dot(texture2D(tex,calcCoordinates( coord + vec2( -w, 0.0))).xyz, vec3(0.299, 0.587, 0.114));
        n[4] = dot(texture2D(tex,calcCoordinates( coord)).xyz, vec3(0.299, 0.587, 0.114));
        n[5] = dot(texture2D(tex,calcCoordinates( coord + vec2(  w, 0.0))).xyz, vec3(0.299, 0.587, 0.114));
        n[6] = dot(texture2D(tex,calcCoordinates( coord + vec2( -w, h))).xyz, vec3(0.299, 0.587, 0.114));
        n[7] = dot(texture2D(tex,calcCoordinates( coord + vec2(0.0, h))).xyz, vec3(0.299, 0.587, 0.114));
        n[8] = dot(texture2D(tex,calcCoordinates( coord + vec2(  w, h))).xyz, vec3(0.299, 0.587, 0.114));
}


void main(void)
{

    vec2 uv = (texcoords);


    ivec2 texSize = textureSize(inputTexture, 0);
    //uv = cubeMapTo2D(vec3(x,y,z));

    float n[9];
    make_kernel( n, inputTexture, uv);

    float sobel_edge_h = n[2] + (2.0*n[5]) + n[8] - (n[0] + (2.0*n[3]) + n[6]);
    float sobel_edge_v = n[0] + (2.0*n[1]) + n[2] - (n[6] + (2.0*n[7]) + n[8]);
    float sobel = sqrt((sobel_edge_h * sobel_edge_h) + (sobel_edge_v * sobel_edge_v));
    float direction = atan(sobel_edge_h, sobel_edge_v)/3.1415926538;
    direction = (direction+1)/2.0f;
    //if(sobel < 0.66){

    //    sobel = 0;
    //}
    fragColor = vec4(vec3(sobel) , 1.0 );
}
