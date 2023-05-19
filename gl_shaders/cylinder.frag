in vec2 texcoords;
out vec4 fragColor;

uniform sampler2D inputTexture;
vec2 cubeMapTo2D(in vec3 cubeVec) {
    float absX = abs(cubeVec.x);
    float absY = abs(cubeVec.y);
    float absZ = abs(cubeVec.z);

    float x, y;

if (absY >= absX && absY >= absZ) { // Y is the biggest
        if (cubeVec.y >= 0) {
            //mitte

            x = cubeVec.x/absY;
            y = -cubeVec.z/absY;
        }
        else {
            //rand
            discard;
        }


    }
else {
    //rand
    discard;
}

    // Map [-1, 1] to [0, 1]
    x = (x + 1.0) * 0.5;
    y = (y + 1.0) * 0.5;

    return vec2(x, y);
 }


void main(void)
{
    float pi = 3.1415926535897932384626433f;
    float fov = 40.0f;
    float x = sin(texcoords.x * pi);
    float y = cos(texcoords.x * pi);
    float z = texcoords.y * radians(fov);
    vec2 uv = cubeMapTo2D(vec3(x,y,z));
    fragColor = vec4(texture(inputTexture, uv).xyz , 1.0 );
        //fragColor = vec4(vec3(1) , 1.0 );
    //fragColor = vec4( sobel.rgb * (1.0-texcoords.y), 1.0 );
}
