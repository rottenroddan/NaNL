//
// Created by steve on 5/26/2023.
//

#ifndef NANL_PAGEDMEMORYBLOCK_CUH
#define NANL_PAGEDMEMORYBLOCK_CUH

#include "BaseMemoryBlock.cuh"
#include "Deleters.cu"

namespace NaNL {

    template<class T>
    class PagedMemoryBlock : public NaNL::Internal::BaseMemoryBlock<T> {
    protected:
        inline explicit PagedMemoryBlock(uint64_t totalSize);
    };

} // NaNL

#include "PagedMemoryBlock.cu"

#endif //NANL_PAGEDMEMORYBLOCK_CUH
