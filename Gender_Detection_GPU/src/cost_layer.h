#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef layer cost_layer;

COST_TYPE get_cost_type(char *s);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer(const cost_layer l, network net);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
