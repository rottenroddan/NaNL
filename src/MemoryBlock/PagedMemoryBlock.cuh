//
// Created by steve on 5/26/2023.
//

#ifndef NANL_PAGEDMEMORYBLOCK_CUH
#define NANL_PAGEDMEMORYBLOCK_CUH

namespace NaNL {

    template<class T>
    class PagedMemoryBlock {
    protected:
        std::unique_ptr<T[], void(*)(T*)> _matrix;

        inline explicit PagedMemoryBlock(uint64_t totalSize);
    };

} // NaNL

#include "PagedMemoryBlock.cu"

#endif //NANL_PAGEDMEMORYBLOCK_CUH
