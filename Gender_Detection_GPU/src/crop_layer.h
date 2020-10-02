#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef layer crop_layer;

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer(const crop_layer l, network net);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif

