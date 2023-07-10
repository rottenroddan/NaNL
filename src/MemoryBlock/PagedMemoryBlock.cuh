//
// Created by steve on 5/26/2023.
//

#ifndef NANL_PAGEDMEMORYBLOCK_CUH
#define NANL_PAGEDMEMORYBLOCK_CUH

#include "HostMemoryBlock.cuh"
#include "Deleters.cu"

namespace NaNL {

    template<typename T, typename Alignment>
    class PagedMemoryBlock : public NaNL::Internal::HostMemoryBlock<T, Alignment> {
    protected:
        inline explicit PagedMemoryBlock(uint64_t rows, uint64_t cols);
    };

} // NaNL

#include "PagedMemoryBlock.cu"

#endif //NANL_PAGEDMEMORYBLOCK_CUH
