//
// Created by steve on 9/20/2023.
//

#include "MatrixInBinaryFileLoader.cuh"

namespace NaNL {

    MatrixInBinaryFileLoader::MatrixInBinaryFileLoader(const std::string& fileName)  {
        // TODO: errno_t eventually
        fopen_s(&this->file, fileName.c_str(), "r");
        //this->file.sync_with_stdio(0);

        if(!this->file) {
            // TODO: error
        }
    }

    template<class T, template<class, class> class Memory, class Alignment>
    Matrix<T, Memory, Alignment> MatrixInBinaryFileLoader::read() {
        uint64_t rows, cols, totalSize;
        uint64_t initBuffer[2];
        size_t readAmount = fread_s(initBuffer, sizeof(initBuffer), sizeof(uint64_t), 2, this->file);

        if(readAmount != sizeof(initBuffer)) {
            // TODO: throw exception.
        }

        rows = initBuffer[0];
        cols = initBuffer[1];
        totalSize = rows * cols;

        Matrix<T, Memory, Alignment> readMatrix(rows, cols);\
        uint64_t bufferSize = totalSize * sizeof(T);

        if(rows == readMatrix.getActualRows() &&
            cols == readMatrix.getActualCols()) {
            fread_s(&readMatrix.getMatrix()[0], totalSize * sizeof(T), sizeof(T), totalSize, this->file);
        } else {
            for(uint64_t i = 0; i < readMatrix.getRows(); i++) {
                fread_s(&readMatrix.getMatrix()[i*readMatrix.getActualCols()], readMatrix.getCols() * sizeof(T), sizeof(T), cols, this->file);
            }
        }

        return readMatrix;
    }

    MatrixInBinaryFileLoader::~MatrixInBinaryFileLoader() {
        fclose(this->file);
    }

    unsigned long MatrixInBinaryFileLoader::getTotalMatrices() {
        return this->totalMatrices;
    }

    unsigned long MatrixInBinaryFileLoader::getCurrentMatrix() {
        return this->currentMatrix;
    }

    void MatrixInBinaryFileLoader::close() {
        if(!closed) {
            fclose(file);
        }
    }
} // NaNL