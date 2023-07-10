//
// Created by steve on 7/8/2023.
//

#include "HostMemoryBlock.cuh"

namespace NaNL {
    namespace Internal {
        template<typename T, typename Alignment>
        HostMemoryBlock<T,Alignment>::HostMemoryBlock(uint64_t rows, uint64_t cols, MemoryTypes memType) :
        BaseMemoryBlock<T, Alignment>(rows, cols, memType) { ; }

        template<typename T, typename Alignment>
        T* HostMemoryBlock<T,Alignment>::operator[](uint64_t i) noexcept {
            return &this->_matrix[i * this->actualCols];
        }

        template<typename T, typename Alignment>
        T HostMemoryBlock<T,Alignment>::get(uint64_t i, uint64_t j) const {
            if(i > this->rows || j > this->cols) {
                // TODO: Throw exception here
                //throw MatrixIndexIsOutOfBounds();
            }
            return this->_matrix[i * this->actualCols + j];
        }
    }
}