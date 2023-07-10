//
// Created by steve on 7/8/2023.
//

#ifndef NANL_HOSTMEMORYBLOCK_CUH
#define NANL_HOSTMEMORYBLOCK_CUH

#include "BaseMemoryBlock.cuh"

namespace NaNL {
    namespace Internal {

        template<typename T, typename Alignment>
        class HostMemoryBlock : public BaseMemoryBlock<T,Alignment> {
        public:
            HostMemoryBlock(uint64_t rows, uint64_t cols, MemoryTypes memType);

            /**
             * Returns a pointer to the startTimepoint of the row based on the index provided.
             * Then you can call a second [] and treat this as your column index.
             * Like such: mxn matrix(M) -> M[3][10] (4th row, 11th column)
             * @param i Index of the startTimepoint of the row you want to access.
             * @return Pointer of type 'T' of the row at the provided index.
             */
            inline T *operator[](uint64_t i) noexcept;

            /**
             * Returns the value associated at the ith and jth position.
             * @param i Index of the row.
             * @param j Index of the col.
             * @return
             */
            inline T get(uint64_t i, uint64_t j) const;
        };
    }
}

#include "HostMemoryBlock.cu"

#endif //NANL_HOSTMEMORYBLOCK_CUH
