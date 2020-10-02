#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>


static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.nweights = c*n*size*size;
    l.nbiases = n;

    l.weights = (float*)calloc(c*n*size*size, sizeof(float));
    l.weight_updates = (float*)calloc(c*n*size*size, sizeof(float));

    l.biases = (float*)calloc(n, sizeof(float));
    l.bias_updates = (float*)calloc(n, sizeof(float));
    //float scale = n/(size*size*c);
    //printf("scale: %f\n", scale);
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    //bilinear_init(l);
    for(i = 0; i < n; ++i){
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);

    l.output = (float*)calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = (float*)calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_deconvolutional_layer;

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
        l.scales = (float*)calloc(n, sizeof(float));
        l.scale_updates = (float*)calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = (float*)calloc(n, sizeof(float));
        l.variance = (float*)calloc(n, sizeof(float));

        l.mean_delta = (float*)calloc(n, sizeof(float));
        l.variance_delta = (float*)calloc(n, sizeof(float));

        l.rolling_mean = (float*)calloc(n, sizeof(float));
        l.rolling_variance = (float*)calloc(n, sizeof(float));
        l.x = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float*)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = (float*)calloc(c*n*size*size, sizeof(float));
        l.v = (float*)calloc(c*n*size*size, sizeof(float));
        l.bias_m = (float*)calloc(n, sizeof(float));
        l.scale_m = (float*)calloc(n, sizeof(float));
        l.bias_v = (float*)calloc(n, sizeof(float));
        l.scale_v = (float*)calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_deconvolutional_layer_gpu;

    if(gpu_index >= 0){

        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(0, n);
            l.variance_gpu = cuda_make_array(0, n);

            l.rolling_mean_gpu = cuda_make_array(0, n);
            l.rolling_variance_gpu = cuda_make_array(0, n);

            l.mean_delta_gpu = cuda_make_array(0, n);
            l.variance_delta_gpu = cuda_make_array(0, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(0, n);

            l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
            l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
        }
    }
    #ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
    #endif
#endif

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void forward_deconvolutional_layer(const layer l, network net)
{
    int i;

    int m = l.size*l.size*l.n;
    int n = l.h*l.w;
    int k = l.c;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = net.input + i*l.c*l.h*l.w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, l.output+i*l.outputs);
    }
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_w*l.out_h);
    }
    activate_array(l.output, l.batch*l.n*l.out_w*l.out_h, l.activation);
}



