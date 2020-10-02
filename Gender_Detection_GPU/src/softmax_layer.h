#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef layer softmax_layer;

softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);

#ifdef GPU
void forward_softmax_layer_gpu(const softmax_layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
