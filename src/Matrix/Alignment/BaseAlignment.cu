//
// Created by steve on 5/26/2023.
//

#include "BaseAlignment.cuh"

namespace NaNL {
    template<class T, template<typename> class Memory>
    BaseAlignment<T, Memory>::BaseAlignment(uint64_t rows, uint64_t cols) : Memory<T>(rows * cols)
    { ; }
}