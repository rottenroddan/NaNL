//
// Created by steve on 12/12/2022.
//

//
// Created by steve on 11/13/2022.
//

#include "BaseMatrix.cuh"

#pragma once

namespace NaNL {

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    BaseMatrix<T, Memory, Alignment>::BaseMatrix(uint64_t rows, uint64_t cols) : Alignment<T, Memory>(rows, cols)  {

    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    T *BaseMatrix<T, Memory, Alignment>::operator[](uint64_t i) noexcept {
        return &this->_matrix[i * this->actualCols];
    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    T BaseMatrix<T, Memory, Alignment>::get(uint64_t i, uint64_t j) const {
        if(i > this->rows || j > this->cols) {
            throw MatrixIndexIsOutOfBounds();
        }
        return this->_matrix[i * this->actualCols + j];
    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    BaseMatrix<T, Memory, Alignment>&
            BaseMatrix<T, Memory, Alignment>::operator=(BaseMatrix<T, Memory, Alignment> &&moveMatrix) noexcept {

        this->align(moveMatrix.getRows(), moveMatrix.getCols());
        moveMatrix.align(0,0);
        this->_matrix = std::move(moveMatrix._matrix);

        return *this;
    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    bool BaseMatrix<T, Memory, Alignment>::validateMatricesAreSameShape(const BaseMatrix<T, Memory, Alignment> &b) const {
        if (this->rows == b.rows && this->cols == b.cols)
            return true;
        return false;
    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    const char *BaseMatrix<T, Memory, Alignment>::MatrixIndexIsOutOfBounds::what() const noexcept {
        return _msg.c_str();
    }

    template<class T, template<typename> class Memory,
            template<class, template<typename > class >class Alignment>
    const char *BaseMatrix<T, Memory, Alignment>::MatrixIsInvalidShape::what() const noexcept {
        return _msg.c_str();
    }
}