#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_upsample_layer(int batch, int w, int h, int c, int stride);
void forward_upsample_layer(const layer l, network net);

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
