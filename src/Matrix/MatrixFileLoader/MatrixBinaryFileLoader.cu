//
// Created by steve on 9/19/2023.
//

#include "MatrixOutBinaryFileLoader.cuh"

namespace NaNL {

    template<typename T>
    MatrixOutBinaryFileLoader<T>::MatrixOutBinaryFileLoader(const std::string& fileName)  {
        // TODO: errno_t eventually
        fopen_s(&this->file, fileName.c_str(), "w+");
        //this->file.sync_with_stdio(0);

        if(!this->file) {
            // TODO: error
        }


    }

    template<class T>
    template<template<class, class> class Memory,
            class Alignment>
    void MatrixOutBinaryFileLoader<T>::loadMatrixIntoFile(Matrix<T, Memory, Alignment> matrix, uint64_t chunkByteSize) {
        // TODO: Potentially more speedups writing in chunks of 2^n... don't know what that will be yet.
        uint64_t rows = matrix.getRows();
        uint64_t cols = matrix.getCols();
        std::fwrite(&rows, sizeof(uint64_t), 1, this->file);
        std::fwrite(&cols, sizeof(uint64_t), 1, this->file);

        if(matrix.getTotalSize() == matrix.getActualTotalSize()) {
            std::fwrite(&matrix.getMatrix()[0], sizeof(T), matrix.getTotalSize(), this->file);
        } else {
            for(uint64_t i = 0; i < matrix.getRows(); i++) {
                std::fwrite(&matrix.getMatrix()[i * matrix.getActualCols()], sizeof(T), matrix.getCols(), this->file);
            }
        }
    }

    template<class T>
    MatrixOutBinaryFileLoader<T>::~MatrixOutBinaryFileLoader() {
        if(!closed) {
            fclose(this->file);
        }
    }

    template<class T>
    unsigned long MatrixOutBinaryFileLoader<T>::getTotalMatrices() {
        return this->totalMatrices;
    }

    template<class T>
    unsigned long MatrixOutBinaryFileLoader<T>::getCurrentMatrix() {
        return this->currentMatrix;
    }

    template<class T>
    void MatrixOutBinaryFileLoader<T>::close() {
        if(!closed) {
            fclose(this->file);
        }
    }
} // NaNL