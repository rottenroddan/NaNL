//
// Created by steve on 5/26/2023.
//

#include "Deleters.cu"
#include "PagedMemoryBlock.cuh"


namespace NaNL {
    template<class T>
    NaNL::PagedMemoryBlock<T>::PagedMemoryBlock(uint64_t totalSize) : _matrix(new T[totalSize], _freePagedMemory)
    {
        std::cout << "Allocating" << std::endl;
    }
}