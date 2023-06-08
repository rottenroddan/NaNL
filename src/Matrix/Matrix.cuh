//
// Created by steve on 12/12/2022.
//
#ifndef NANL_MATRIX_CUH
#define NANL_MATRIX_CUH

#include "../BaseMatrix/BaseMatrix.cuh"
#include "../Kernels/MatrixKernels.cuh"
#include "../ThreadPool/ThreadPool.cuh"
#include "../CudaUtil/CudaUtil.cuh"
#include <utility>
#include <mma.h>

namespace NaNL {

    enum class MatrixDeviceOperation { Host, Cuda, TensorCores };

    /**
     * Forward declaration that this class will exist.
     */
    class MatrixUtility;


    template<class T, template<typename> class Memory = NaNL::PagedMemoryBlock,
            template<class, template<typename> class> class Alignment = NaNL::Unaligned>
    class Matrix : public BaseMatrix<T, Memory, Alignment> {
    protected:
        friend class MatrixUtility;
    public:
        inline Matrix(uint64_t rows, uint64_t cols);
        inline Matrix(const Matrix<T, Memory, Alignment> &copyMatrix) noexcept;
        inline Matrix(Matrix<T, Memory, Alignment> &&copyMatrix) noexcept;
        inline Matrix<T, Memory, Alignment> &operator=(const Matrix<T, Memory, Alignment> &rhs);

        template<template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
        inline Matrix<T, rMemory, rAlignment> add(const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Host);
        template<template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
        inline Matrix<T, rMemory, rAlignment> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Host);

    };

//    template<class T>
//    class Matrix<T,PagedMemoryBlock,Unaligned> : public BaseMatrix<T, PagedMemoryBlock, Unaligned> {
//    public:
//        inline Matrix(uint64_t rows, uint64_t cols);
//
//        inline void add(const BaseMatrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Host);
//        //inline void add(const BaseMatrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Cuda);
//    };
//
//    template<class T>
//    class Matrix<T,PinnedMemoryBlock,Unaligned> : public BaseMatrix<T, PinnedMemoryBlock, Unaligned> {
//    public:
//        inline Matrix(uint64_t rows, uint64_t cols);
//
//        inline void add(const BaseMatrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Cuda);
//        //inline void add(const BaseMatrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Cuda);
//    };

//    template<typename T>
//    class Matrix<T,Device::Host> : virtual public BaseMatrix<T> {
//    private:
//        inline void _allocateMemory(unsigned long rows, unsigned long cols);
//        inline void _hostAddMatrices(T* _a, T* _b, T* _c, unsigned long _blockSize, unsigned long _offset);
//    public:
//        inline void add(const BaseMatrix<T> &b) override;
//
//        inline Matrix(const Matrix<T, Device::Host> &copyMatrix) noexcept;
//        inline Matrix(Matrix<T, Device::Host> &&copyMatrix) noexcept;
//        inline Matrix(unsigned long numberOfRows, unsigned long numberOfCols);
//        inline Matrix<T, Device::Host> &operator=(const Matrix<T, Device::Host> &rhs);
//    };
//
//
//    template<typename T>
//    class Matrix<T,Device::Cuda> : virtual public BaseMatrix<T> {
//    private:
//        inline void _allocateMemory(unsigned long rows, unsigned long cols);
//    public:
//        inline void add(const BaseMatrix<T> &b) override;
//        inline Matrix(const Matrix<T, Device::Cuda> &copyMatrix) noexcept;
//        inline Matrix(Matrix<T, Device::Cuda> &&copyMatrix) noexcept;
//        inline Matrix(unsigned long numberOfRows, unsigned long numberOfCols);
//        inline Matrix<T, Device::Cuda> &operator=(const Matrix<T, Device::Cuda> &rhs);
//    };
}

#include "MatrixUtility/MatrixUtility.cuh"
#include "MatrixPaged/MatrixHost.cu"
#include "MatrixPinned/MatrixCuda.cu"
//#include "MatrixDevice/MatrixDevice.cu"

#endif //NANL_MATRIX_CUH
