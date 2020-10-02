#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = (float*)calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}
