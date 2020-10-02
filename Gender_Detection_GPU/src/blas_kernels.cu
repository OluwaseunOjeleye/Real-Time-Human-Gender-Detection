#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}


__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= N) return;
	int f = (index / spatial) % filters;

	x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
}


extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

__global__ void copy_kernel(int N, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
	axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
}

extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
	axpy_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
	check_error(cudaPeekAtLastError());
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
	pow_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX, Y, INCY);
	check_error(cudaPeekAtLastError());
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY] *= X[i*INCX];
}

extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
	mul_kernel << <cuda_gridsize(N), BLOCK >> > (N, X, INCX, Y, INCY);
	check_error(cudaPeekAtLastError());
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] = ALPHA;
}

extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
	fill_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] = ALPHA;
}

extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
{
	const_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
    //out[out_index] += add[add_index];
}

extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
    check_error(cudaPeekAtLastError());
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] *= ALPHA;
}

extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
	scal_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

__global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] += ALPHA;
}

extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
{
	add_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float diff = truth[i] - pred[i];
		float abs_val = fabsf(diff);
		if (abs_val < 1) {
			error[i] = diff * diff;
			delta[i] = diff;
		}
		else {
			error[i] = 2 * abs_val - 1;
			delta[i] = (diff > 0) ? 1 : -1;
		}
	}
}

extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	smooth_l1_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void l2_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float diff = truth[i] - pred[i];
		error[i] = diff * diff; //I know this is technically wrong, deal with it.
		delta[i] = diff;
	}
}

extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	l2_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float diff = truth[i] - pred[i];
		error[i] = abs(diff);
		delta[i] = (diff > 0) ? 1 : -1;
	}
}

extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	l1_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void wgan_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		error[i] = truth[i] ? -pred[i] : pred[i];
		delta[i] = (truth[i] > 0) ? 1 : -1;
	}
}

extern "C" void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	wgan_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void supp_kernel(int N, float ALPHA, float *X, int INCX)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) {
		if ((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
	}
}

extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
{
	supp_kernel << <cuda_gridsize(N), BLOCK >> > (N, ALPHA, X, INCX);
	check_error(cudaPeekAtLastError());
}

__global__ void mask_kernel(int n, float *x, float mask_num, float *mask, float val)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n && mask[i] == mask_num) x[i] = val;
}

extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
{
	mask_kernel << <cuda_gridsize(N), BLOCK >> > (N, X, mask_num, mask, val);
	check_error(cudaPeekAtLastError());
}

__global__ void scale_mask_kernel(int n, float *x, float mask_num, float *mask, float scale)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n && mask[i] == mask_num) x[i] *= scale;
}

extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
{
	scale_mask_kernel << <cuda_gridsize(N), BLOCK >> > (N, X, mask_num, mask, scale);
	check_error(cudaPeekAtLastError());
}

__global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float t = truth[i];
		float p = pred[i];
		error[i] = (t) ? -log(p) : 0;
		delta[i] = t - p;
	}
}

extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	softmax_x_ent_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void logistic_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		float t = truth[i];
		float p = pred[i];
		error[i] = -t * log(p + .0000001) - (1 - t)*log(1 - p + .0000001);
		delta[i] = t - p;
	}
}

extern "C" void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
	logistic_x_ent_kernel << <cuda_gridsize(n), BLOCK >> > (n, pred, truth, delta, error);
	check_error(cudaPeekAtLastError());
}

__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int in_s = i % spatial;
	i = i / spatial;
	int in_c = i % layers;
	i = i / layers;
	int b = i;

	int i1 = b * layers*spatial + in_c * spatial + in_s;
	int i2 = b * layers*spatial + in_s * layers + in_c;

	if (forward) out[i2] = x[i1];
	else out[i1] = x[i2];
}

extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
	int size = spatial * batch*layers;
	flatten_kernel << <cuda_gridsize(size), BLOCK >> > (size, x, spatial, layers, batch, forward, out);
	check_error(cudaPeekAtLastError());
}

__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int in_index = i;
	int in_w = i % w;
	i = i / w;
	int in_h = i % h;
	i = i / h;
	int in_c = i % c;
	i = i / c;
	int b = i % batch;

	int out_c = c / (stride*stride);

	int c2 = in_c % out_c;
	int offset = in_c / out_c;
	int w2 = in_w * stride + offset % stride;
	int h2 = in_h * stride + offset / stride;
	//printf("%d\n", offset);
	int out_index = w2 + w * stride*(h2 + h * stride*(c2 + out_c * b));

	// printf("%d %d %d\n", w2, h2, c2);
	 //printf("%d %d\n", in_index, out_index);
	 //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

	if (forward) out[out_index] = x[in_index];
	else out[in_index] = x[out_index];
	//if(forward) out[1] = x[1];
	//else out[0] = x[0];
}

extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
	int size = w * h*c*batch;
	reorg_kernel << <cuda_gridsize(size), BLOCK >> > (size, x, w, h, c, batch, stride, forward, out);
	check_error(cudaPeekAtLastError());
}

__global__ void l2norm_kernel(int N, float *x, float *dx, int batch, int filters, int spatial)
{
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= N) return;
	int b = index / spatial;
	int i = index % spatial;
	int f;
	float sum = 0;
	for (f = 0; f < filters; ++f) {
		int index = b * filters*spatial + f * spatial + i;
		sum += powf(x[index], 2);
	}
	sum = sqrtf(sum);
	if (sum == 0) sum = 1;
	//printf("%f\n", sum);
	for (f = 0; f < filters; ++f) {
		int index = b * filters*spatial + f * spatial + i;
		x[index] /= sum;
		dx[index] = (1 - x[index]) / sum;
	}
}

extern "C" void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial)
{
	size_t N = batch * spatial;
	l2norm_kernel << <cuda_gridsize(N), BLOCK >> > (N, x, dx, batch, filters, spatial);
	check_error(cudaPeekAtLastError());
}

__global__ void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = s[i] * a[i] + (1 - s[i])*(b ? b[i] : 0);
	}
}

extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
	weighted_sum_kernel << <cuda_gridsize(num), BLOCK >> > (num, a, b, s, c);
	check_error(cudaPeekAtLastError());
}

__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i) {
		int val = input[i*stride];
		largest = (val > largest) ? val : largest;
	}
	for (i = 0; i < n; ++i) {
		float e = expf(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}

__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= batch * groups) return;
	int b = id / groups;
	int g = id % groups;
	softmax_device(input + b * batch_offset + g * group_offset, n, temp, stride, output + b * batch_offset + g * group_offset);
}

extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	softmax_kernel << <cuda_gridsize(batch*groups), BLOCK >> > (input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
	check_error(cudaPeekAtLastError());
}

__global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
	int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= spatial * batch*groups) return;
	int s = id % spatial;
	id = id / spatial;
	int g = id % groups;
	int b = id / groups;
	int goff = group_offset[g] * spatial;
	int boff = b * stride;
	softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

extern "C" void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier)
{
	int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
	int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
	/*
	   static int *tree_groups_size = 0;
	   static int *tree_groups_offset = 0;
	   if(!tree_groups_size){
	   tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
	   tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
	   }
	 */
	int num = spatial * batch*hier.groups;
	softmax_tree_kernel << <cuda_gridsize(num), BLOCK >> > (input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
	check_error(cudaPeekAtLastError());
	cuda_free((float *)tree_groups_size);
	cuda_free((float *)tree_groups_offset);
}

__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
	size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int out_index = i;
	int out_w = i % (w*stride);
	i = i / (w*stride);
	int out_h = i % (h*stride);
	i = i / (h*stride);
	int out_c = i % c;
	i = i / c;
	int b = i % batch;

	int in_w = out_w / stride;
	int in_h = out_h / stride;
	int in_c = out_c;

	int in_index = b * w*h*c + in_c * w*h + in_h * w + in_w;


	if (forward) out[out_index] += scale * x[in_index];
	else atomicAdd(x + in_index, scale * out[out_index]);
}
extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
	size_t size = w * h*c*batch*stride*stride;
	upsample_kernel << <cuda_gridsize(size), BLOCK >> > (size, in, w, h, c, batch, stride, forward, scale, out);
	check_error(cudaPeekAtLastError());
}