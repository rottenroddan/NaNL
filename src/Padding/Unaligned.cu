//
// Created by steve on 5/26/2023.
//

#include "Unaligned.cuh"

namespace NaNL {
    Unaligned::Unaligned(uint64_t rows, uint64_t cols) : BaseAlignment(rows, cols) {

        align(rows, cols);
    }
} // NaNL