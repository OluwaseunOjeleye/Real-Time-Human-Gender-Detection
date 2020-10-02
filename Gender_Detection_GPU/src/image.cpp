#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void save_image_options(image im, const char *name, IMTYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
    save_image_options(im, name, JPG, 80);
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(char *filename, int w, int h, int c)
{
    image out = load_image_stb(filename, c);
    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
	float ratio = ((float)x / max) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	//printf("%f\n", r);
	return r;
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}