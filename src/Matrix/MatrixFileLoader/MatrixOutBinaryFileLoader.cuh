//
// Created by steve on 9/19/2023.
//

#ifndef NANL_MATRIXOUTBINARYFILELOADER_CUH
#define NANL_MATRIXOUTBINARYFILELOADER_CUH

#include "../../BaseMatrix/BaseMatrix.cuh"
#include "../Matrix.cuh"

namespace NaNL {

    template<typename T>
    class MatrixOutBinaryFileLoader {
    private:
        unsigned long totalMatrices = 0;
        unsigned long currentMatrix = 0;
        bool closed = false;
        FILE* file;
    public:
        inline MatrixOutBinaryFileLoader() = delete;
        explicit inline MatrixOutBinaryFileLoader(const std::string& file);

        inline unsigned long getTotalMatrices();
        inline unsigned long getCurrentMatrix();

        template<template<class, class> class Memory = NaNL::PagedMemoryBlock,
                class Alignment = NaNL::Unaligned>
        inline void loadMatrixIntoFile(Matrix<T, Memory, Alignment> matrix, uint64_t chunkByteSize = 0xffff);

        inline ~MatrixOutBinaryFileLoader();

        inline void close();
    };

} // NaNL

#include "MatrixBinaryFileLoader.cu"

#endif //NANL_MATRIXOUTBINARYFILELOADER_CUH
