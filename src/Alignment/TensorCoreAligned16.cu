//
// Created by steve on 7/31/2023.
//

#include "TensorCoreAligned16.cuh"

NaNL::TensorCoreAligned16::TensorCoreAligned16(uint64_t rows, uint64_t cols) : Internal::TensorCoreAligned128Bits(rows, cols, ALIGNMENT_MULTIPLE_OF){ ; }