//
// Created by steve on 11/27/2022.
//
#include "Matrix.cuh"

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
NaNL::Matrix<T, Memory, Alignment>::Matrix(uint64_t rows, uint64_t cols) : BaseMatrix<T, Memory, Alignment>(rows, cols)
{

}

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
NaNL::Matrix<T, Memory, Alignment>::Matrix(const Matrix<T, Memory, Alignment> &copyMatrix) noexcept : BaseMatrix<T, Memory, Alignment>(copyMatrix.getRows(), copyMatrix.getCols())
{
    for(uint64_t i = 0; i < this->rows; i++) {
        for(uint64_t j = 0; j < this->cols; j++) {
            this->operator[](i)[j] = copyMatrix.get(i,j);
        }
    }
}

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
NaNL::Matrix<T, Memory, Alignment>::Matrix(Matrix<T, Memory, Alignment> &&copyMatrix) noexcept : BaseMatrix<T, Memory, Alignment>(copyMatrix.rows, copyMatrix.cols)
{
    this->_matrix = std::move(copyMatrix._matrix);
}

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
NaNL::Matrix<T, Memory, Alignment>&
NaNL::Matrix<T, Memory, Alignment>::Matrix::operator=(const Matrix <T, Memory, Alignment> &rhs) {
    return *this;
}

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
template<template<typename> class rMemory,
        template<class, template<typename> class> class rAlignment>
NaNL::Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::add(const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device)
{
    return NaNL::MatrixUtility::add<T, rMemory, rAlignment>((const Matrix<T,Memory, Alignment>&)(*this), (const Matrix<T,Memory, Alignment>&)b, device);
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





