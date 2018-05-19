__kernel void circle(
    __constant float *C,
    __write_only image2d_t color_image,
    __write_only image2d_t velocity_image,
    float2 center, float radius, 
    float4 color, float4 velocity
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float2 r = convert_float2(p) - center;
    if (dot(r, r) < radius*radius) {
        write_imagef(color_image, p, color);
        write_imagef(velocity_image, p, velocity);
    }
}
