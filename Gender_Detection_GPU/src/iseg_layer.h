#ifndef ISEG_LAYER_H
#define ISEG_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_iseg_layer(int batch, int w, int h, int classes, int ids);
void forward_iseg_layer(const layer l, network net);

#ifdef GPU
void forward_iseg_layer_gpu(const layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
