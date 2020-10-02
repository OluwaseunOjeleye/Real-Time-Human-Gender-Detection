#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void forward_deconvolutional_layer(const layer l, network net);

#ifdef GPU
void forward_deconvolutional_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif

