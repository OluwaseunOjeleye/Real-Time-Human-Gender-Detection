#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "utils.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "yolo_layer.h"
#include "batchnorm_layer.h"
#include "shortcut_layer.h"
#include "parser.h"

network *make_network(int n)
{
    network *net = (network*)calloc(1, sizeof(network));
    net->n = n;
    net->layers = (layer*)calloc(net->n, sizeof(layer));
    net->seen = (size_t*)calloc(1, sizeof(size_t));
    net->t    = (int*)calloc(1, sizeof(int));
    net->cost = (float*)calloc(1, sizeof(float));
    return net;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}

void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}


float * network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = (float*)calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
	network net = *netp;
	cuda_set_device(net.gpu_index);
	cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
	if (net.truth) {
		cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
	}

	int i;
	for (i = 0; i < net.n; ++i) {
		net.index = i;
		layer l = net.layers[i];
		if (l.delta_gpu) {
			fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
		}
		l.forward_gpu(l, net);
		net.input_gpu = l.output_gpu;
		net.input = l.output;
		if (l.truth) {
			net.truth_gpu = l.output_gpu;
			net.truth = l.output;
		}
	}
	pull_network_output(netp);
	calc_network_cost(netp);
}

void pull_network_output(network *net)
{
	layer l = get_network_output_layer(net);
	cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif