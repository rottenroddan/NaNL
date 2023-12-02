//
// Created by steve on 8/1/2023.
//

#include "TensorCoreAligned32.cuh"

NaNL::TensorCoreAligned32::TensorCoreAligned32(uint64_t rows, uint64_t cols) : Internal::TensorCoreAligned128Bits(rows, cols, ALIGNMENT_MULTIPLE_OF) { ; }