#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	
typedef layer avgpool_layer;

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void forward_avgpool_layer(const avgpool_layer l, network net);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif
	
#ifdef __cplusplus
}
#endif

#endif

