//
// Created by steve on 5/26/2023.
//

#include "BaseAlignment.cuh"

namespace NaNL {
    BaseAlignment::BaseAlignment(uint64_t rows, uint64_t cols, uint64_t multiple) {
        BaseAlignment::align(rows, cols, multiple);
    }

    uint64_t BaseAlignment::getRows() const {
        return this->rows;
    }

    uint64_t BaseAlignment::getCols() const {
        return this->cols;
    }

    uint64_t BaseAlignment::getTotalSize() const {
        return this->totalSize;
    }

    uint64_t BaseAlignment::getActualRows() const {
        return this->actualRows;
    }

    uint64_t BaseAlignment::getActualCols() const {
        return this->actualCols;
    }

    uint64_t BaseAlignment::getActualTotalSize() const {
        return this->actualTotalSize;
    }

    void BaseAlignment::align(uint64_t rows, uint64_t cols, uint64_t multiple) {
        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows * cols;

        // alignment for actual matrix size if needed.
        uint64_t rowRemainder = rows % multiple;
        this->actualRows = rows + (rowRemainder == 0 ? 0 : (multiple - rowRemainder));
        uint64_t colRemainder = rows % multiple;
        this->actualCols = cols + (colRemainder == 0 ? 0 : (multiple - colRemainder));
        this->actualTotalSize = actualRows * actualCols;
    }
}