#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void forward_region_layer(const layer l, network net);

#ifdef GPU
void forward_region_layer_gpu(const layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
