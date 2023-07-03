//
// Created by steve on 5/26/2023.
//


#include "PagedMemoryBlock.cuh"


namespace NaNL {
    template<class T>
    PagedMemoryBlock<T>::PagedMemoryBlock(uint64_t totalSize) :
    Internal::BaseMemoryBlock<T>(Internal::MemoryTypes::Host)
    {
        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(new T[totalSize], _freePagedMemory);
    }
}