//
// Created by steve on 12/12/2022.
//
#ifndef NANL_MATRIX_CUH
#define NANL_MATRIX_CUH

#include "../BaseMatrix/BaseMatrix.cuh"
#include "../Kernels/MatrixKernels.cuh"
#include "../ThreadPool/ThreadPool.cuh"
#include <utility>
#include <mma.h>


namespace NaNL {

    enum class MatrixAddOperation {
        Host, Cuda, CudaStride, TensorCores
    };

    template<typename T, template<typename, typename> class U, class V>
    concept IsDerivedFromHostMemoryBlock = std::is_base_of_v<NaNL::Internal::HostMemoryBlock<T, V>, U<T, V>>;

    template<class T, template<class, class> class Memory = NaNL::PagedMemoryBlock,
            class Padding = NaNL::Unaligned>
    class Matrix : public BaseMatrix<T, Memory, Padding> {
    protected:


        mutable std::shared_mutex _shared_mutex;
    public:
//        template<template<class, class> class uMemory, class uPadding,
//                template<class, class> class rMemory, class rPadding>
//        static void _addMatricesOnCuda(const Matrix<T, Memory, Padding> &a,
//                                       const Matrix<T, uMemory, uPadding> &b,
//                                       Matrix<T, rMemory, rPadding> &c) requires
//        IsDerivedFromDeviceMemoryBlock<T, Memory, Padding> &&
//        IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding> &&
//        IsDerivedFromDeviceMemoryBlock<T, rMemory, rPadding>;

        inline Matrix(uint64_t rows, uint64_t cols);

        inline Matrix(const Matrix<T, Memory, Padding> &copyMatrix) noexcept;

        inline Matrix(Matrix<T, Memory, Padding> &&copyMatrix) noexcept;

        inline Matrix<T, Memory, Padding> &operator=(const Matrix<T, Memory, Padding> &rhs);

        inline Matrix<T, Memory, Padding> &operator=(Matrix<T, Memory, Padding> &&rhs) noexcept = default;

        inline std::shared_mutex& getMutex() const;

        template<template<class, class> class rMemory, class rPadding, class R = T>
        inline Matrix<R, rMemory, rPadding> copyTo() const;

//        template<template<class, class> class rMemory, class rPadding, class R = T>
//        inline Matrix<R, rMemory, rPadding> copyToCast() const;

        template<template<class, class> class rMemory, class rPadding>
        inline Matrix<T, rMemory, rPadding> moveTo() const &&;

        template<template<class, class> class rMemory, class rPadding,
                template<class, class> class uMemory, class uPadding>
        inline Matrix<T, rMemory, rPadding>
        add(const Matrix<T, uMemory, uPadding> &b, MatrixAddOperation device = MatrixAddOperation::Host);
//        template<template<typename> class rMemory, template<class, template<typename> class> class rPadding>
//        inline Matrix<T, rMemory, rPadding> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device = MatrixDeviceOperation::Host);

        template<template<class, class> class rMemory, class rPadding,
                template<class, class> class uMemory, class uPadding>
        static Matrix<T, rMemory, rPadding>
        addHost(const Matrix<T, Memory, Padding> &a, const Matrix<T, uMemory, uPadding> &b);

        template<template<class, class> class rMemory, class rPadding,
                template<class, class> class uMemory, class uPadding>
        static Matrix<T, rMemory, rPadding>
        addCuda(const Matrix<T, Memory, Padding> &a, const Matrix<T, uMemory, uPadding> &b);

//        Matrix <T, rMemory, rPadding>
//        addCuda(const Matrix <T, Memory, Padding> &a, const Matrix <T, uMemory, uPadding> &b);
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

#include "../MatrixUtility/MatrixTypeTraits.cuh"
#include "MatrixUtility/MatrixUtility.cuh"
#include "MatrixOperations.cu"
#include "Matrix.cu"

#endif //NANL_MATRIX_CUH
