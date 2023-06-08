//
// Created by steve on 5/29/2023.
//

#ifndef NANL_PINNEDMEMORYBLOCK_CUH
#define NANL_PINNEDMEMORYBLOCK_CUH

#include "../CudaUtil/CudaUtil.cuh"

namespace NaNL {

    template<class T>
    class PinnedMemoryBlock {
    protected:
        std::unique_ptr<T[], void(*)(T*)> _matrix;

        inline explicit PinnedMemoryBlock(uint64_t totalSize);
    };

} // NaNL

#include "PinnedMemoryBlock.cu"

#endif //NANL_PINNEDMEMORYBLOCK_CUH
