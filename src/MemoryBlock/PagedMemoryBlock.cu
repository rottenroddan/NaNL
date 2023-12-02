//
// Created by steve on 5/26/2023.
//

#include <new>
#include "PagedMemoryBlock.cuh"

namespace NaNL {

    template<class T, class Alignment>
    PagedMemoryBlock<T, Alignment>::PagedMemoryBlock(uint64_t rows, uint64_t cols) :
    Internal::HostMemoryBlock<T, Alignment>(rows, cols, Internal::MemoryTypes::Host)
    {
        //T* tempPtr = new T[this->actualRows * this->actualCols];
        //this->_matrix = std::unique_ptr<T[], void (*)(T*)>(tempPtr, _freePagedMemory);

        size_t alignmentBytes = std::hardware_constructive_interference_size;
        T* tempPtr = static_cast<T*>(_aligned_malloc(sizeof (T) * this->actualRows * this->actualCols, alignmentBytes));

        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(tempPtr, _freePagedMemory);
    }
}