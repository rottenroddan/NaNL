//
// Created by steve on 5/26/2023.
//

#include "Unaligned.cuh"

namespace NaNL {
    template<class T, template<typename> class Memory>
    Unaligned<T, Memory>::Unaligned(uint64_t rows, uint64_t cols) : BaseAlignment<T, Memory>(rows, cols) {

        align(rows, cols);
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getRows() const {
        return this->rows;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualRows() const {
        return this->actualRows;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getCols() const {
        return this->cols;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualCols() const {
        return this->actualCols;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getTotalSize() const {
        return this->totalSize;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualTotalSize() const {
        return this->actualTotalSize;
    }

    template<class T, template<typename> class Memory>
    void Unaligned<T, Memory>::align(uint64_t rows, uint64_t cols) {
        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows * cols;
        this->actualRows = rows;
        this->actualCols = cols;
        this->actualTotalSize = this->totalSize;
    }
} // NaNL