//
// Created by steve on 5/29/2023.
//

#ifndef NANL_PINNEDMEMORYBLOCK_CUH
#define NANL_PINNEDMEMORYBLOCK_CUH

#include "../CudaUtil/CudaUtil.cuh"
#include "HostMemoryBlock.cuh"
#include "Deleters.cu"

namespace NaNL {

    template<typename T, typename Alignment>
    class PinnedMemoryBlock : public NaNL::Internal::HostMemoryBlock<T, Alignment> {
    protected:
        inline explicit PinnedMemoryBlock(uint64_t rows, uint64_t cols);
    };

} // NaNL

#include "PinnedMemoryBlock.cu"

#endif //NANL_PINNEDMEMORYBLOCK_CUH
