//
// Created by steve on 5/4/2023.
//
#include "Matrix.cuh"

#pragma once

template<class T, template<typename> class Memory,
        template<class, template<typename> class> class Alignment>
template<template<typename> class rMemory,
        template<class, template<typename> class> class rAlignment>
NaNL::Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::add(const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device)
{
    return NaNL::MatrixUtility::add<T, rMemory, rAlignment>((const Matrix<T,Memory, Alignment>&)(*this), (const Matrix<T,Memory, Alignment>&)b, device);
}

//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Cuda>::Matrix(unsigned long numberOfRows, unsigned long numberOfCols) :
//        BaseMatrix<T>(numberOfRows, numberOfCols, _freePinnedMemory) {
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
//NaNL::Matrix<T, NaNL::Device::Cuda>::Matrix(const NaNL::Matrix<T, NaNL::Device::Cuda> &copyMatrix) noexcept :
//        BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePinnedMemory) { ; }
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Cuda>::Matrix(NaNL::Matrix<T, NaNL::Device::Cuda> &&copyMatrix)  noexcept :
//        BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePinnedMemory) {
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
//void NaNL::Matrix<T, NaNL::Device::Cuda>::_allocateMemory(unsigned long rows, unsigned long cols)
//{
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    T *_pinnedArr;
//    gpuErrchk(cudaMallocHost((void **) &_pinnedArr, this->totalSize * sizeof(T)));
//    this->matrix = std::unique_ptr<T[], void (*)(T*)>(_pinnedArr, _freePinnedMemory);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Cuda>::add(const NaNL::BaseMatrix<T> &bMatrix) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//    //TODO: comment better.
//
//    unsigned long mSize = this->totalSize;
//    T *a = static_cast<T*>(this->matrix.get());
//    T *b = static_cast<T*>(bMatrix.matrix.get());
//
//    // device array pointers.
//    T *dev_a, *dev_b;
//
//    // Allocate memory for
//    gpuErrchk(cudaMalloc((void**) &dev_a, mSize * sizeof(T)));
//    gpuErrchk(cudaMalloc((void**) &dev_b, mSize * sizeof(T)));
//
//    gpuErrchk(cudaMemcpy(dev_a, a,  mSize * sizeof(T), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(dev_b, b,  mSize * sizeof(T), cudaMemcpyHostToDevice));
//
//    dim3 threadsPerBlock(32, 32, 1);
//    dim3 numBlocks(mSize / (threadsPerBlock.x) + 1);
//
//    deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_a, mSize);
//
//    gpuErrchk(cudaPeekAtLastError());
//    gpuErrchk(cudaDeviceSynchronize());
//
//    // Copy the data from device a to host a.
//    gpuErrchk(cudaMemcpy(a, dev_a, mSize * sizeof(T), cudaMemcpyDeviceToHost));
//
//    // Free Memory in Device.
//    gpuErrchk(cudaFree(dev_a));
//    gpuErrchk(cudaFree(dev_b));
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Cuda> &
//NaNL::Matrix<T, NaNL::Device::Cuda>::operator=(const NaNL::Matrix<T, NaNL::Device::Cuda> &rhs) {
//    return *this;
//}