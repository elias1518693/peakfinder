in vec2 texcoords;
out vec4 fragColor;

uniform sampler2D inputTexture;


void make_kernel(inout float n[9], sampler2D tex, vec2 coord)
{
        vec2 texSize = textureSize(tex, 0);
        float w = 1.0 / texSize.x;
        float h = 1.0 / texSize.y;
        coord += vec2(w,h);

        n[0] = dot(texture2D(tex, coord + vec2( -w, -h)).xyz, vec3(0.299, 0.587, 0.114));
        n[1] = dot(texture2D(tex, coord + vec2(0.0, -h)).xyz, vec3(0.299, 0.587, 0.114));
        n[2] = dot(texture2D(tex, coord + vec2(  w, -h)).xyz, vec3(0.299, 0.587, 0.114));
        n[3] = dot(texture2D(tex, coord + vec2( -w, 0.0)).xyz, vec3(0.299, 0.587, 0.114));
        n[4] = dot(texture2D(tex, coord).xyz, vec3(0.299, 0.587, 0.114));
        n[5] = dot(texture2D(tex, coord + vec2(  w, 0.0)).xyz, vec3(0.299, 0.587, 0.114));
        n[6] = dot(texture2D(tex, coord + vec2( -w, h)).xyz, vec3(0.299, 0.587, 0.114));
        n[7] = dot(texture2D(tex, coord + vec2(0.0, h)).xyz, vec3(0.299, 0.587, 0.114));
        n[8] = dot(texture2D(tex, coord + vec2(  w, h)).xyz, vec3(0.299, 0.587, 0.114));
}

void main(void)
{
        vec2 uv = vec2(texcoords.x, 1.0 - texcoords.y);
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
        //fragColor = vec4( sobel.rgb * (1.0-texcoords.y), 1.0 );
}
