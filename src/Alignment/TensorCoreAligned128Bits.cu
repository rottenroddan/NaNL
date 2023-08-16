//
// Created by steve on 8/1/2023.
//

#include "TensorCoreAligned128Bits.cuh"

namespace NaNL::Internal {
    TensorCoreAligned128Bits::TensorCoreAligned128Bits(uint64_t rows, uint64_t cols) : BaseAlignment(rows, cols) { ; }
}
