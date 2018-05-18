#define TIME_STEP 0
#define GRID_SIZE 1
#define VISCOSITY 2


int flatten(int2 p, int2 s) {
    return p.x + s.x*p.y;
}

bool is_inside(int2 p, int2 s) {
    return 0 <= p.x && p.x < s.x && 0 <= p.y && p.y < s.y;
}

#define LAPLACIAN(n) \
float##n laplacian##n( \
    __global const float *field, \
    int2 p, int2 s, float grid_size \
) { \
    float##n r = 0.0; \
    float##n v = vload##n(flatten(p, s), field); \
    const int2 nd[4] = {(int2)(1,0),(int2)(0,1),(int2)(-1,0),(int2)(0,-1)}; \
    int i; \
    for (i = 0; i < 4; ++i) { \
        int2 np = p + nd[i]; \
        float##n nv; \
        if (is_inside(np, s)) { \
            nv = vload##n(flatten(np, s), field); \
        } else { \
            nv = v; \
        } \
        r += nv - v; \
    } \
    return r/(grid_size*grid_size); \
}

LAPLACIAN()
LAPLACIAN(2)
LAPLACIAN(3)
LAPLACIAN(4)

    
__kernel void put_circle(
    __constant float *C,
    __global float *color_buffer,
    __global float *velocity_buffer,
    float2 center, float radius, 
    float4 color, float2 velocity
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 r = convert_float2(p) - center;
    if (dot(r, r) < radius*radius) {
        vstore2(velocity, flatten(p, s), velocity_buffer);
        vstore4(color, flatten(p, s), color_buffer);
    }
}

/*
__kernel void advect(
    __constant float *C,
    __global const float *src,
    __global       float *dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float4 v = vload4(flatten(p, s), color_src);
    v += C[COLOR_DIFFISION]*delta4(color_src, p, s);
    vstore4(v, flatten(p, s), color_dst);
}
*/

__kernel void diffuse(
    __constant float *C,
    __global const float *velocity_buffer_src,
    __global       float *velocity_buffer_dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 v = vload2(flatten(p, s), velocity_buffer_src);
    float2 dv = laplacian2(velocity_buffer_src, p, s, C[GRID_SIZE]);
    v += dv*C[VISCOSITY]*C[TIME_STEP];
    vstore2(v, flatten(p, s), velocity_buffer_dst);
}


__kernel void render(
    __constant float *C,
    __global const float *color_buffer,
    __global const float *velocity_buffer,
    __global       uchar *screen
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float4 cf = vload4(flatten(p, s), color_buffer);
    cf.xy += vload2(flatten(p, s), velocity_buffer);
    cf = clamp(cf, 0.0f, 1.0f);
    uchar3 cb = convert_uchar3(255.0f*cf.xyz);
    vstore3(cb, flatten(p, s), screen);
}
