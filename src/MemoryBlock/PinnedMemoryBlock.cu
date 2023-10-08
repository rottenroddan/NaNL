//
// Created by steve on 5/29/2023.
//

#include "PinnedMemoryBlock.cuh"

namespace NaNL {
    template<class T, class Alignment>
    PinnedMemoryBlock<T, Alignment>::PinnedMemoryBlock(uint64_t rows, uint64_t cols) :
    Internal::HostMemoryBlock<T, Alignment>(rows, cols, Internal::MemoryTypes::CudaPinned)
    {
        T *_pinnedArr;
        gpuErrchk(cudaMallocHost((void **) &_pinnedArr, this->actualRows * this->actualCols * sizeof(T)));
        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(_pinnedArr, _freePinnedMemory);
    }
} // NaNL