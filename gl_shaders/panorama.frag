in highp vec2 texcoords;
uniform sampler2D texture_sampler0;
uniform sampler2D texture_sampler1;
uniform sampler2D texture_sampler2;
uniform sampler2D texture_sampler3;
uniform sampler2D texture_sampler4;
uniform sampler2D texture_sampler5;

out lowp vec4 out_Color;

vec2 cubeMapTo2D(in vec3 cubeVec) {
    float absX = abs(cubeVec.x);
    float absY = abs(cubeVec.y);
    float absZ = abs(cubeVec.z);

    float x, y;

    if (absX >= absY && absX >= absZ) { // X is the biggest
        if (cubeVec.x > 0.0) {
            x = -cubeVec.z/absX;
            y = -cubeVec.y/absX;
        }
        else {
            x = cubeVec.z/absX;
            y = -cubeVec.y/absX;
        }

    }
    else if (absY >= absX && absY >= absZ) { // Y is the biggest
        if (cubeVec.y > 0.0) {
            x = cubeVec.x/absY;
            y = cubeVec.z/absY;
        }
        else {
            x = cubeVec.x/absY;
            y = -cubeVec.z/absY;
        }

    }
    else { // Z is the biggest
        if (cubeVec.z > 0.0) {
            x = cubeVec.x/absZ;
            y = -cubeVec.y/absZ;
        }
        else {
            x = -cubeVec.x/absZ;
            y = -cubeVec.y/absZ;
        }

    }

    // Map [-1, 1] to [0, 1]
    x = (x + 1.0) * 0.5;
    y = (y + 1.0) * 0.5;

    return vec2(x, y);
 }

sampler2D getSampler(in vec3 cubeVec) {
    float absX = abs(cubeVec.x);
    float absY = abs(cubeVec.y);
    float absZ = abs(cubeVec.z);

    if (absX >= absY && absX >= absZ) { // X is the biggest
        if (cubeVec.x > 0.0) {
            return texture_sampler0;
        }
        else {
            return texture_sampler1;
        }
    }
    else if (absY >= absX && absY >= absZ) { // Y is the biggest
        if (cubeVec.y > 0.0) {
            return texture_sampler2;
        }
        else {
            return texture_sampler3;
        }
    }
    else { // Z is the biggest
        if (cubeVec.z > 0.0) {
            return texture_sampler4;
        }
        else {
            return texture_sampler5;
        }
    }
}

void main() {

    float fov = 90.0;
    float pi = 3.1415926535897932384626433f;


    float x = cos(texcoords.x * pi);
    float z = texcoords.y * tan(fov/2.0);
    float y = sin(texcoords.x * pi);

    out_Color = texture(getSampler(vec3(x,y,z)), cubeMapTo2D(normalize(vec3(x,y,z))));
    //out_Color = vec4(cubeMapTo2D(vec3(x,y,z)),0,1);
}
