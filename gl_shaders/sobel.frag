in vec2 texcoords;
out vec4 fragColor;

uniform sampler2D inputTexture;


void main()
{
    vec2 textureSize = textureSize(inputTexture, 0);
    vec2 texelSize = 1.0 / textureSize;

    // Sobel filter kernel
    float kernel[9] = float[9](-1, 0, 1,
                                -2, 0, 2,
                                -1, 0, 1);

    vec3 edgeColors[9];

    for (int i = 0; i < 9; i++) {
        vec2 offset = texelSize * vec2(i % 3 - 1, i / 3.0f - 1.0f) + vec2(texelSize);
        edgeColors[i] = textureLod(inputTexture, texcoords + offset, 0.0).rgb;
    }

    vec3 edgeX = kernel[0] * edgeColors[0] + kernel[1] * edgeColors[1] + kernel[2] * edgeColors[2]
               + kernel[3] * edgeColors[3] + kernel[4] * edgeColors[4] + kernel[5] * edgeColors[5]
               + kernel[6] * edgeColors[6] + kernel[7] * edgeColors[7] + kernel[8] * edgeColors[8];

    vec3 edgeY = kernel[0] * edgeColors[0] + kernel[3] * edgeColors[3] + kernel[6] * edgeColors[6]
               + kernel[1] * edgeColors[1] + kernel[4] * edgeColors[4] + kernel[7] * edgeColors[7]
               + kernel[2] * edgeColors[2] + kernel[5] * edgeColors[5] + kernel[8] * edgeColors[8];

    float edgeMagnitude = length(vec2(length(edgeX), length(edgeY)));
    //if(edgeMagnitude < 2){
    //    edgeMagnitude = 0;
    //}
    fragColor = vec4(vec3(edgeMagnitude), 1.0);
}
