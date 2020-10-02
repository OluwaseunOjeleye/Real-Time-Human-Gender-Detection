
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif


layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);
void forward_crnn_layer(layer l, network net);

#ifdef GPU
void forward_crnn_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif

