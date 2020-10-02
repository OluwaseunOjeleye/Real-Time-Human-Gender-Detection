#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);
void forward_connected_layer(layer l, network net);

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
void push_connected_layer(layer l);
#endif

#endif

