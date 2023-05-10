//
// Created by steve on 12/24/2022.
//

#ifndef NANL_MATRIXKERNELS_CUH
#define NANL_MATRIXKERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>


namespace NaNL {
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    template<typename T>
    __global__ void deviceAddMatrices(T *_dev_a, T *_dev_b, T *_dev_c, unsigned long mSize);
}




#include "MatrixKernels.cu"

#endif //NANL_MATRIXKERNELS_CUH
