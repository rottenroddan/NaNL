//
// Created by steve on 11/27/2022.
//
#include "Matrix.cuh"

namespace NaNL {
    template<class T, template<class, class> class Memory,
            class Alignment>
    Matrix<T, Memory, Alignment>::Matrix()
            : BaseMatrix<T, Memory, Alignment>(0, 0) {

    }

    template<class T, template<class, class> class Memory,
            class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(uint64_t rows, uint64_t cols)
            : BaseMatrix<T, Memory, Alignment>(rows, cols) {

    }

    template<class T, template<class, class> class Memory, class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(const Matrix<T, Memory, Alignment> &copyMatrix) noexcept
            : BaseMatrix<T, Memory, Alignment>(copyMatrix.getRows(), copyMatrix.getCols()) {
        for (uint64_t i = 0; i < this->rows; i++) {
            for (uint64_t j = 0; j < this->cols; j++) {
                this->operator[](i)[j] = copyMatrix.get(i, j);
            }
        }
    }

    template<class T, template<class, class> class Memory, class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(Matrix<T, Memory, Alignment> &&copyMatrix) noexcept
            : BaseMatrix<T, Memory, Alignment>(copyMatrix.rows, copyMatrix.cols) {
        this->_matrix = std::move(copyMatrix._matrix);
    }

    template<class T, template<class, class> class Memory, class Alignment>
    NaNL::Matrix<T, Memory, Alignment> &
    NaNL::Matrix<T, Memory, Alignment>::Matrix::operator=(const Matrix<T, Memory, Alignment> &rhs) {
        return *this;
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment>
    Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::copyTo() const {

        /*
         * TODO: Clean this method up, a lot of redundancy
         */
        Matrix<T, rMemory, rAlignment> copyMatrix(this->getRows(), this->getCols());

        if constexpr(std::is_base_of_v<Matrix<T, PagedMemoryBlock, rAlignment>, Matrix<T, rMemory, rAlignment>>
                        && std::is_base_of_v<Matrix<T, PagedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>) {
            // host to host
            if (Internal::MemoryTypes::Host == copyMatrix.getMemoryType() &&
                Internal::MemoryTypes::Host == this->getMemoryType()) {
                for (uint64_t i = 0; i < this->getRows(); i++) {
                    for (uint64_t j = 0; j < this->getCols(); j++) {
                        copyMatrix[i][j] = this->get(i, j);
                    }
                }

                return copyMatrix;
            }
        }

        // default
        cudaMemcpyKind memcpyKind = cudaMemcpyHostToHost;

        // pinned to pinned, or pinned to paged / paged to pinned.
        if(Internal::MemoryTypes::CudaPinned == copyMatrix.getMemoryType() &&
            (Internal::MemoryTypes::Host == this->getMemoryType() || Internal::MemoryTypes::CudaPinned == this->getMemoryType()) ||
            Internal::MemoryTypes::CudaPinned == this->getMemoryType() &&
            (Internal::MemoryTypes::Host == copyMatrix.getMemoryType() || Internal::MemoryTypes::CudaPinned == copyMatrix.getMemoryType())) {
            //gpuErrchk(cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getTotalSize() * sizeof(T), cudaMemcpyHostToHost));
            memcpyKind = cudaMemcpyHostToHost;
        }

        // host to device
        if(Internal::MemoryTypes::CudaDevice == copyMatrix.getMemoryType() &&
            (Internal::MemoryTypes::Host == this->getMemoryType() || Internal::MemoryTypes::CudaPinned == this->getMemoryType())) {
            for(uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T), cudaMemcpyHostToDevice));
            }
            memcpyKind = cudaMemcpyHostToDevice;
        }

        // device to host
        if((Internal::MemoryTypes::Host == copyMatrix.getMemoryType() || Internal::MemoryTypes::Host == copyMatrix.getMemoryType()) &&
            Internal::MemoryTypes::CudaDevice == this->getMemoryType() ) {
            for(uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T), cudaMemcpyDeviceToHost));
            }
            memcpyKind = cudaMemcpyDeviceToHost;
        }

        // device to device
        if((Internal::MemoryTypes::CudaDevice == copyMatrix.getMemoryType() && Internal::MemoryTypes::CudaDevice == copyMatrix.getMemoryType())) {
            for(uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            memcpyKind = cudaMemcpyDeviceToDevice;
        }

        // Perform copy based on kind of copy decided from above.
        // If the actual total size(for different alignments) is
        // the same, we can perform a 1:1 copy rather a loop-based
        // copy over.
        if(this->getActualRows() == copyMatrix.getActualRows()
            && this->getActualCols() == copyMatrix.getActualCols()
            && this->getActualTotalSize() == copyMatrix.getActualTotalSize()) {
            cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T), memcpyKind);
        } else {
            for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                     this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T),
                                     memcpyKind));
            }
        }

        return copyMatrix;
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment>
    Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::moveTo() const {
        return copyTo<rMemory, rAlignment>();
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment,
            template<class, class> class uMemory, class uAlignment>
    Matrix<T, rMemory, rAlignment>
    Matrix<T, Memory, Alignment>::add(const Matrix<T, uMemory, uAlignment> &b, MatrixDeviceOperation device) {

        if (MatrixDeviceOperation::TensorCores == device) {

        } else if (MatrixDeviceOperation::Cuda == device) {

        } else /*Host*/ {
            return NaNL::Internal::MatrixUtility::addHost
                    <T, rMemory, rAlignment>((const Matrix<T, Memory, Alignment> &) (*this), b);
        }
    }
}


//template<class T>
//NaNL::Matrix<T,NaNL::PagedMemoryBlock, NaNL::Unaligned>::Matrix(uint64_t rows, uint64_t cols) : BaseMatrix<T, NaNL::PagedMemoryBlock, NaNL::Unaligned>(rows, cols)
//{
//
//}
//
//
//template<class T>
//void NaNL::Matrix<T,NaNL::PagedMemoryBlock,NaNL::Unaligned>::add(const BaseMatrix <T, NaNL::PagedMemoryBlock, NaNL::Unaligned> &b, NaNL::MatrixDeviceOperation device) {
//    std::cout << "The one we want." << std::endl;
//}
//
//template<class T>
//void NaNL::Matrix<T,NaNL::PinnedMemoryBlock,NaNL::Unaligned>::add(const BaseMatrix <T, NaNL::PinnedMemoryBlock, NaNL::Unaligned> &b, NaNL::MatrixDeviceOperation device) {
//    std::cout << "The one we don't want." << std::endl;
//}

//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(unsigned long numberOfRows, unsigned long numberOfCols) :
//        BaseMatrix<T>(numberOfRows, numberOfCols, _freePagedMemory) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    _allocateMemory(numberOfRows, numberOfCols);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(const NaNL::Matrix<T, NaNL::Device::Host> &copyMatrix) noexcept :
//BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePagedMemory) { ; }
//
//
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(NaNL::Matrix<T, NaNL::Device::Host> &&copyMatrix)  noexcept :
//BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePagedMemory) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    this->matrix = std::move(copyMatrix.matrix);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::_allocateMemory(unsigned long rows, unsigned long cols)
//{
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    T* _pagedArr = new T[this->totalSize];
//    this->matrix = std::unique_ptr<T[], void(*)(T*)>(_pagedArr, _freePagedMemory);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::_hostAddMatrices(T* _a, T* _b, T* _c, unsigned long _blockSize, unsigned long _offset) {
//    for(unsigned long i = 0; i < _blockSize; i++) {
//        _c[_offset + i] = _a[_offset + i] + _b[_offset + i];
//    }
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::add(const NaNL::BaseMatrix<T> &bMatrix) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    if(this->validateMatricesAreSameShape(bMatrix)) {
//        //throw new MatrixIsInvalidShape();
//    }
//
//    T* a = this->matrix.get();
//    T* b = bMatrix.matrix.get();
//    T* c = this->matrix.get();
//
//    // get total totalThreads.
//    unsigned long totalThreads = NaNL::ThreadPool::getInstance()->getAllocatedThreads();
//
//    // calculate block size per thread.
//    unsigned long blockSize = this->totalSize / totalThreads;
//    unsigned long remainder = this->totalSize - blockSize * totalThreads;
//    unsigned long threadOffset = 0;
//
//    NaNL::ThreadPool* threadPool = NaNL::ThreadPool::getInstance();
//    std::deque<std::future<void>> results;
//
//    for(unsigned long i = 0; i < totalThreads; i++) {
//        unsigned long modifiedBlockSize = (remainder == 0) ? blockSize : blockSize + 1;
//        std::future<void> result = threadPool->queue([this, a, b, c, modifiedBlockSize, threadOffset] { _hostAddMatrices(a,b,c, modifiedBlockSize, threadOffset); });
//        results.push_back(std::move(result));
//        threadOffset += modifiedBlockSize;
//
//        if(remainder != 0) {
//            remainder--;
//        }
//    }
//
//    // wait on all futures to be populated/deferred.
//    for(auto & result : results) {
//        result.wait();
//    }
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host> &
//NaNL::Matrix<T, NaNL::Device::Host>::operator=(const NaNL::Matrix<T, NaNL::Device::Host> &rhs) {
//    return *this;
//}





