//
// Created by steve on 5/26/2023.
//

#include "Unaligned.cuh"

namespace NaNL {
    Unaligned::Unaligned(uint64_t rows, uint64_t cols) : BaseAlignment(rows, cols) {

        align(rows, cols);
    }

    uint64_t Unaligned::Unaligned::getRows() const {
        return this->rows;
    }

    uint64_t Unaligned::Unaligned::getActualRows() const {
        return this->actualRows;
    }

    uint64_t Unaligned::Unaligned::getCols() const {
        return this->cols;
    }

    uint64_t Unaligned::Unaligned::getActualCols() const {
        return this->actualCols;
    }

    uint64_t Unaligned::Unaligned::getTotalSize() const {
        return this->totalSize;
    }

    uint64_t Unaligned::Unaligned::getActualTotalSize() const {
        return this->actualTotalSize;
    }

    void Unaligned::align(uint64_t rows, uint64_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows * cols;
        this->actualRows = rows;
        this->actualCols = cols;
        this->actualTotalSize = this->totalSize;
    }
} // NaNL