#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef layer local_layer;

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);
void forward_local_layer(const local_layer layer, network net);

#ifdef GPU
void forward_local_layer_gpu(local_layer layer, network net);
void push_local_layer(local_layer layer);
#endif

#ifdef __cplusplus
}
#endif

#endif

