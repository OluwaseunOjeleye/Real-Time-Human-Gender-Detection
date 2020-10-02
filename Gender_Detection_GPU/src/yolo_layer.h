#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_yolo_layer(const layer l, network net);

int yolo_num_detections(layer l, float thresh);

#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
