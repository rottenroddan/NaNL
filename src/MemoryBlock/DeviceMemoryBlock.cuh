//
// Created by steve on 6/10/2023.
//

#ifndef NANL_DEVICEMEMORYBLOCK_CUH
#define NANL_DEVICEMEMORYBLOCK_CUH

#include "../CudaUtil/CudaUtil.cuh"
#include "BaseMemoryBlock.cuh"
#include "Deleters.cu"

#include <cuda.h>
#include <cuda_runtime.h>

namespace NaNL {

    template<typename T, typename Alignment>
    class DeviceMemoryBlock : public Internal::BaseMemoryBlock<T, Alignment> {
    protected:
        inline explicit DeviceMemoryBlock(uint64_t rows, uint64_t cols);
    };

} // NaNL

#include "DeviceMemoryBlock.cu"

#endif //NANL_DEVICEMEMORYBLOCK_CUH
