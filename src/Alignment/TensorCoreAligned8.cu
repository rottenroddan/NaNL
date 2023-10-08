//
// Created by steve on 7/31/2023.
//

#include "TensorCoreAligned8.cuh"

NaNL::TensorCoreAligned8::TensorCoreAligned8(uint64_t rows, uint64_t cols) : Internal::TensorCoreAligned128Bits(rows, cols, ALIGNMENT_MULTIPLE_OF){ ; }
