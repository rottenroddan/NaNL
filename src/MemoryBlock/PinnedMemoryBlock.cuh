//
// Created by steve on 5/29/2023.
//

#ifndef NANL_PINNEDMEMORYBLOCK_CUH
#define NANL_PINNEDMEMORYBLOCK_CUH

#include "../CudaUtil/CudaUtil.cuh"
#include "Deleters.cu"

namespace NaNL {

    template<class T>
    class PinnedMemoryBlock : public NaNL::Internal::BaseMemoryBlock<T> {
    protected:
        inline explicit PinnedMemoryBlock(uint64_t totalSize);
    };

} // NaNL

#include "PinnedMemoryBlock.cu"

#endif //NANL_PINNEDMEMORYBLOCK_CUH
