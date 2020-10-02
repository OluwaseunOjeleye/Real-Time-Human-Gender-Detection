#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void forward_reorg_layer(const layer l, network net);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif

