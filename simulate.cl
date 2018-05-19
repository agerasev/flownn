#define TIME_STEP 0
#define GRID_SIZE 1
#define VISCOSITY 2


float divergence2(
    __read_only image2d_t field_image,
    int2 p, float grid_size
) {
    return (
        + read_imagef(field_image, p + (int2)( 1, 0)).x
        - read_imagef(field_image, p + (int2)(-1, 0)).x
        + read_imagef(field_image, p + (int2)( 0, 1)).y
        - read_imagef(field_image, p + (int2)( 0,-1)).y
    )/(2.0f*grid_size);
}

float2 gradient2(
    __read_only image2d_t field_image,
    int2 p, float grid_size
) {
    return (float2)(
        + read_imagef(field_image, p + (int2)( 1, 0)).x
        - read_imagef(field_image, p + (int2)(-1, 0)).x
        ,
        + read_imagef(field_image, p + (int2)( 0, 1)).x
        - read_imagef(field_image, p + (int2)( 0,-1)).x
    )/grid_size;
}

float jacobi(
    __read_only image2d_t field_image,
    int2 p, float a, float b, float value
) {
    return (
        + read_imagef(field_image, p + (int2)( 1, 0)).x
        + read_imagef(field_image, p + (int2)( 0, 1)).x
        + read_imagef(field_image, p + (int2)(-1, 0)).x
        + read_imagef(field_image, p + (int2)( 0,-1)).x
        + a*value
    )/b;
}

float2 jacobi2(
    __read_only image2d_t field_image,
    int2 p, float a, float b, float2 value
) {
    return (
        + read_imagef(field_image, p + (int2)( 1, 0)).xy
        + read_imagef(field_image, p + (int2)( 0, 1)).xy
        + read_imagef(field_image, p + (int2)(-1, 0)).xy
        + read_imagef(field_image, p + (int2)( 0,-1)).xy
        + a*value
    )/b;
}

float2 laplacian(
    __read_only image2d_t field_image,
    int2 p, float grid_size
) {
    return jacobi(
        field_image, p, 
        -4.0f, grid_size*grid_size,
        read_imagef(field_image, p).x
    );
}

float2 laplacian2(
    __read_only image2d_t field_image,
    int2 p, float grid_size
) {
    return jacobi2(
        field_image, p, 
        -4.0f, grid_size*grid_size,
        read_imagef(field_image, p).xy
    );
}


__kernel void advect(
    __constant float *C,
    
    __read_only  image2d_t color_image_src,
    __write_only image2d_t color_image_dst,
    
    __read_only  image2d_t velocity_image_src,
    __write_only image2d_t velocity_image_dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
    
    float2 v = read_imagef(velocity_image_src, p).xy;
    float2 ap = convert_float2(p) + (float2)(0.5,0.5) - C[TIME_STEP]*v.xy;
    
    float4 ac = read_imagef(color_image_src, sampler, ap);
    float4 av = read_imagef(velocity_image_src, sampler, ap);
    write_imagef(velocity_image_dst, p, av);
    write_imagef(color_image_dst, p, ac);
}

__kernel void diffuse(
    __constant float *C,
    __read_only  image2d_t velocity_image_src,
    __write_only image2d_t velocity_image_dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 v = read_imagef(velocity_image_src, p).xy;
    float a = (C[GRID_SIZE]*C[GRID_SIZE])/(C[VISCOSITY]*C[TIME_STEP]);
    float b = 4.0f + a;
    float2 w = jacobi2(velocity_image_src, p, a, b, v);
    write_imagef(velocity_image_dst, p, (float4)(w,0,1));
}

__kernel void pressure_compute_step(
    __constant float *C,
    __read_only  image2d_t velocity_image,
    __read_only  image2d_t pressure_image_src,
    __write_only image2d_t pressure_image_dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float v = divergence2(velocity_image, p, C[GRID_SIZE]);
    
    float a = -C[GRID_SIZE]*C[GRID_SIZE];
    float b = 4.0f;
    float w = jacobi(pressure_image_src, p, a, b, v);
    
    write_imagef(pressure_image_dst, p, (float4)(w,0,0,1));
}

__kernel void pressure_subtract_gradient(
    __constant float *C,
    __read_only  image2d_t velocity_image_src,
    __write_only image2d_t velocity_image_dst,
    __read_only  image2d_t pressure_image_src,
    __write_only image2d_t pressure_image_dst
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 dv = gradient2(pressure_image_src, p, C[GRID_SIZE]);
    float2 v = read_imagef(velocity_image_src, p).xy;
    write_imagef(velocity_image_dst, p, (float4)(v - dv,0,1));
    write_imagef(pressure_image_dst, p, (float4)(0,0,0,1));
}
