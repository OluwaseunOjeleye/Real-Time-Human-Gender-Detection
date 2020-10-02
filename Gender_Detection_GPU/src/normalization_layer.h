#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void forward_normalization_layer(const layer layer, network net);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
