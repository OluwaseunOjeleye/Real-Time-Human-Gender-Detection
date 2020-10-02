
#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
void forward_gru_layer(layer l, network state);

#ifdef GPU
void forward_gru_layer_gpu(layer l, network state);
#endif

#ifdef __cplusplus
}
#endif

#endif

