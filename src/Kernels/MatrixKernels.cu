//
// Created by steve on 12/24/2022.
//
#include "MatrixKernels.cuh"

template<typename T>
__global__ void NaNL::Internal::Kernels::deviceAddMatrices(const T *_dev_a, const T *_dev_b, T *_dev_c, uint64_t mSize) {
    uint64_t tId = blockIdx.x * blockDim.x + threadIdx.x;

    if(tId < mSize) {
        _dev_c[tId] = _dev_a[tId] + _dev_b[tId];
    }
}

template<typename T>
__global__ void NaNL::Internal::Kernels::deviceAddMatricesWithOffset(const T *_dev_a, const T *_dev_b, T *_dev_c,
                                                                     uint64_t mSize,
                                                                     uint64_t rowSize,
                                                                     uint64_t colSize,
                                                                     uint64_t aOffset,
                                                                     uint64_t bOffset,
                                                                     uint64_t cOffset) {

    uint64_t tId = blockIdx.x * blockDim.x + threadIdx.x;

//    __syncthreads();
//    if(tId == 0) {
//        printf("sizeof: %llu, mSize: %llu, rowSize: %llu, colSize: %llu, aOffset: %llu, bOffset: %llu, cOffset: %llu \n",sizeof(uint64_t),mSize, rowSize, colSize, aOffset, bOffset, cOffset);
//    }

    if(tId < mSize) {
//        printf("tId: %llu, c: %llu, a: %llu, b: %llu \n", tId, ((tId / colSize) * cOffset  + tId % colSize), ((tId / colSize) * aOffset + tId % colSize), ((tId / colSize) * bOffset + tId % colSize));
        _dev_c[(tId / colSize) * cOffset  + tId % colSize] = _dev_a[(tId / colSize) * aOffset + tId % colSize] + _dev_b[(tId / colSize) * bOffset + tId % colSize];
    }
}

template<typename T, typename U>
__global__ void NaNL::Internal::Kernels::deviceMatrixCast(T *_dev_a, U *_dev_b, unsigned long mSize) {
    unsigned long tId = blockIdx.x * blockDim.x + threadIdx.x;

    if(tId < mSize) {
        _dev_b[tId] = (U) _dev_a[tId];
    }
}