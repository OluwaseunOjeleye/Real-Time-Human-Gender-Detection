#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

#ifdef __cplusplus
extern "C" {
#endif
	extern int gpu_index;

	void check_error(cudaError_t status);
	cublasHandle_t blas_handle();
	int *cuda_make_int_array(int *x, size_t n);
	void cuda_random(float *x_gpu, size_t n);
	float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
	dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
	cudnnHandle_t cudnn_handle();
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#else // GPU
//LIB_API void cuda_set_device(int n);
#endif // GPU
#endif // DARKCUDA_H
