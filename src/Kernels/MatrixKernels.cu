//
// Created by steve on 12/24/2022.
//
#include "MatrixKernels.cuh"

template<typename T>
extern __global__ void NaNL::Internal::Kernels::deviceAddMatrices(const T *_dev_a, const T *_dev_b, T *_dev_c, unsigned long mSize) {
    unsigned long tId = blockIdx.x * blockDim.x + threadIdx.x;

    if(tId < mSize) {
        _dev_c[tId] = _dev_a[tId] + _dev_b[tId];
    }
}

template<typename T, typename U>
extern __global__ void NaNL::Internal::Kernels::deviceMatrixCast(T *_dev_a, U *_dev_b, unsigned long mSize) {
    unsigned long tId = blockIdx.x * blockDim.x + threadIdx.x;

    if(tId < mSize) {
        _dev_b[tId] = (U) _dev_a[tId];
    }
}