#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network net);

#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
