//
// Created by steve on 12/24/2022.
//
#include "MatrixKernels.cuh"

template<typename T>
extern __global__ void NaNL::deviceAddMatrices(T *_dev_a, T *_dev_b, T *_dev_c, unsigned long mSize) {
    unsigned long tId = blockIdx.x * blockDim.x + threadIdx.x;

    if(tId < mSize) {
        _dev_c[tId] = _dev_a[tId] + _dev_b[tId];
    }
}