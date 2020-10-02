#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef layer convolutional_layer;

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(convolutional_layer *l);

void add_bias(float *output, float *biases, int batch, int n, int size);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void push_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);

#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif