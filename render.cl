__kernel void render(
    __read_only  image2d_t color_image,
    __read_only  image2d_t velocity_image,
    __read_only  image2d_t pressure_image,
    __write_only image2d_t screen
) {
    int2 p = (int2)(get_global_id(0), get_global_id(1));
    int2 s = (int2)(get_global_size(0), get_global_size(1));
    
    float4 cf = read_imagef(color_image, p);
    uint4 cb = convert_uint4(255.0f*cf);
    cb.w = 255;
    write_imageui(screen, p, cb);
}
