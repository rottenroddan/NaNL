//
// Created by steve on 12/12/2022.
//

//
// Created by steve on 11/13/2022.
//

#include "BaseMatrix.cuh"

#pragma once

namespace NaNL {


    /**
     * Function that frees paged memory from the pointer provided.
     * @tparam T Pointer type.
     * @param ptr_ Pointer to the address to free.
     */
    template<typename T>
    void _freePagedMemory(T* ptr_) { delete [] ptr_;}

    /**
     * Function that frees pinned memory from the pointer provided.
     * Calls <code>cudaFreeHost</code> to free pinned memory.
     * @tparam T Pointer type.
     * @param ptr_ Pointer to the address to free.
     */
    template<typename T>
    void _freePinnedMemory(T* ptr_) { cudaFreeHost(ptr_);}


    template<typename T>
    BaseMatrix<T>::BaseMatrix(unsigned long numberOfRows, unsigned long numberOfCols, void (*deleter)(T *)) :
            matrix(nullptr, deleter) {
#ifdef PERFORMANCE_LOGGING
        PERFORMANCE_LOGGING_START;
#endif

        _setDimensions(numberOfRows, numberOfCols);

#ifdef PERFORMANCE_LOGGING
        PERFORMANCE_LOGGING_END;
#endif
    }

    template<typename T>
    void BaseMatrix<T>::_setDimensions(unsigned long numberOfRows, unsigned long numberOfCols) {
        rows = numberOfRows;
        cols = numberOfCols;
        totalSize = numberOfRows * numberOfCols;
    }

    template<typename T>
    T *BaseMatrix<T>::operator[](unsigned long i) noexcept {
        return &matrix[i * cols];
    }

    template<typename T>
    T BaseMatrix<T>::get(unsigned int i, unsigned int j) const {
        if(i > this->rows || j > this->cols) {
            throw MatrixIndexIsOutOfBounds();
        }
        return matrix[i * cols + j];
    }

    template<typename T>
    unsigned long BaseMatrix<T>::getTotalSize() const {
        return totalSize;
    }

    template<typename T>
    unsigned long BaseMatrix<T>::getRows() const {
        return rows;
    }

    template<typename T>
    unsigned long BaseMatrix<T>::getCols() const {
        return cols;
    }

    template<typename T>
    bool BaseMatrix<T>::validateMatricesAreSameShape(const BaseMatrix<T> &b) const {
        if (this->rows == b.rows && this->cols == b.cols)
            return true;
        return false;
    }

    template<typename T>
    const char *BaseMatrix<T>::MatrixIndexIsOutOfBounds::what() const noexcept {
        return _msg.c_str();
    }

    template<typename T>
    const char *BaseMatrix<T>::MatrixIsInvalidShape::what() const noexcept {
        return _msg.c_str();
    }
}