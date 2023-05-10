//
// Created by steve on 12/12/2022.
//
#ifndef NANL_MATRIX_CUH
#define NANL_MATRIX_CUH

#include "BaseMatrix.cuh"
#include "../Kernels/MatrixKernels.cuh"
#include "../ThreadPool/ThreadPool.cuh"
#include <utility>
#include <mma.h>

namespace NaNL {

    template<typename T>
    class Matrix<T,Device::Host> : virtual public BaseMatrix<T> {
    private:
        inline void _allocateMemory(unsigned long rows, unsigned long cols);
        inline void _hostAddMatrices(T* _a, T* _b, T* _c, unsigned long _blockSize, unsigned long _offset);
    public:
        inline void add(const BaseMatrix<T> &b) override;

        inline Matrix(const Matrix<T, Device::Host> &copyMatrix) noexcept;
        inline Matrix(Matrix<T, Device::Host> &&copyMatrix) noexcept;
        inline Matrix(unsigned long numberOfRows, unsigned long numberOfCols);
        inline Matrix<T, Device::Host> &operator=(const Matrix<T, Device::Host> &rhs);
    };


    template<typename T>
    class Matrix<T,Device::Cuda> : virtual public BaseMatrix<T> {
    private:
        inline void _allocateMemory(unsigned long rows, unsigned long cols);
    public:
        inline void add(const BaseMatrix<T> &b) override;
        inline Matrix(const Matrix<T, Device::Cuda> &copyMatrix) noexcept;
        inline Matrix(Matrix<T, Device::Cuda> &&copyMatrix) noexcept;
        inline Matrix(unsigned long numberOfRows, unsigned long numberOfCols);
        inline Matrix<T, Device::Cuda> &operator=(const Matrix<T, Device::Cuda> &rhs);
    };
}

#include "MatrixHost.cu"
#include "MatrixCuda.cu"

#endif //NANL_MATRIX_CUH
