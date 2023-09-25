//
// Created by steve on 5/23/2023.
//

#ifndef NANL_CUDAUTIL_CUH
#define NANL_CUDAUTIL_CUH

namespace NaNL {
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
}

#endif //NANL_CUDAUTIL_CUH
