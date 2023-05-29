//
// Created by steve on 5/26/2023.
//

#include "Unaligned.cuh"

namespace NaNL {
    template<class T, template<typename> class Memory>
    Unaligned<T, Memory>::Unaligned(uint64_t rows, uint64_t cols) : BaseAlignment<T, Memory>(rows, cols) {
        std::cout << rows << " : " << cols << std::endl;

        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows * cols;
        this->actualRows = rows;
        this->actualCols = cols;
        this->actualTotalSize = this->totalSize;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getRows() {
        return this->rows;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualRows() {
        return this->actualRows;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getCols() {
        return this->cols;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualCols() {
        return this->actualCols;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getTotalSize() {
        return this->totalSize;
    }

    template<class T, template<typename> class Memory>
    uint64_t Unaligned<T, Memory>::Unaligned::getActualTotalSize() {
        return this->actualTotalSize;
    }
} // NaNL