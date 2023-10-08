//
// Created by steve on 12/24/2022.
//

#ifndef NANL_MATRIXKERNELS_CUH
#define NANL_MATRIXKERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <mma.h>


namespace NaNL::Internal::Kernels {
    template<typename T>
    __global__ void deviceAddMatrices(const T *_dev_a, const T *_dev_b, T *_dev_c, uint64_t mSize);

    template<typename T>
    __global__ void
    deviceAddMatricesWithOffset(const T *_dev_a, const T *_dev_b, T *_dev_c, uint64_t mSize,
                                uint64_t rowSize, uint64_t colSize,
                                uint64_t aOffset, uint64_t bOffset,
                                uint64_t cOffset);

    template<typename T, typename U>
    __global__ void deviceMatrixCast(T *_dev_a, U *_dev_b, unsigned long mSize);
}




#include "MatrixKernels.cu"

#endif //NANL_MATRIXKERNELS_CUH
