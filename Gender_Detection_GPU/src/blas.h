#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

//reorg-layer
void flatten(float *x, int size, int layers, int batch, int forward);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

//normalization-layer
void const_cpu(int N, float ALPHA, float *X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

//gru, lstm and normalization-layer
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

//shortcut-layer
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

//batchnorm-layer
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void scale_bias(float *output, float *scales, int batch, int n, int size);

//l2norm-layer
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

//cost-layer
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);

//logistic-layer
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

//softmax-layer
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

//gru-layer
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

//detection-layer
void softmax(float *input, int n, float temp, int stride, float *output);

//region and softmax-layer
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

//upsample-layer
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);


#ifdef GPU
#include "cuda.h"
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, float * X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
void const_gpu(int N, float ALPHA, float *X, int INCX);

void supp_gpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);

void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);

void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void fill_gpu(int N, float ALPHA, float * X, int INCX);

void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);

void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out);

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);
void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#endif

#ifdef __cplusplus
}
#endif
#endif
