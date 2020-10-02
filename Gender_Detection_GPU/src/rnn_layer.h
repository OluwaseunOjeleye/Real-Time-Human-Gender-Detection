
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

#ifdef __cplusplus
extern "C" {
#endif

layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam);
void forward_rnn_layer(layer l, network net);


#ifdef GPU
void forward_rnn_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif

