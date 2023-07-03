//
// Created by steve on 5/29/2023.
//

#include "PinnedMemoryBlock.cuh"

namespace NaNL {
    template<class T>
    PinnedMemoryBlock<T>::PinnedMemoryBlock(uint64_t totalSize) :
    Internal::BaseMemoryBlock<T>(Internal::MemoryTypes::CudaPinned)
    {
        T *_pinnedArr;
        gpuErrchk(cudaMallocHost((void **) &_pinnedArr, totalSize * sizeof(T)));
        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(_pinnedArr, _freePinnedMemory);
    }
} // NaNL