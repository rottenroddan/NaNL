//
// Created by steve on 12/17/2022.
//

#include "MatrixFileLoader.cuh"

MatrixFileLoader::MatrixFileLoader(const std::string& file)  {
    this->file.open(file);

    if(this->file.fail()) {
        // TODO: provide functionality
        return;
    }

    std::string firstLine;

    // first line should describe how many matrices.
    std::getline(this->file, firstLine);

    std::stringstream ss(firstLine);

    // first thing to read in this file is total matrices to expect.
    ss >> this->totalMatrices;
}

template<typename T, typename U>
NaNL::Matrix<T, U> MatrixFileLoader::readValuesIntoMatrix() {
    // get first line from file.
    std::string line;
    std::getline(this->file, line);

    std::stringstream ss(line);
    unsigned long rows, cols;

    ss >> rows;
    ss >> cols;

    NaNL::Matrix<T, U> matrixFromFile(rows, cols);

    // for each number of rows, get line to populate matrix per row.
    for(unsigned long i = 0; i < rows; i++) {
        std::getline(this->file, line);
        ss.str(line); // next line.
        ss.clear();
        for(unsigned long j = 0; j < cols; j++) {
            T tempVal;
            ss >> tempVal;
            matrixFromFile[i][j] = tempVal;
        }
    }

    return matrixFromFile;
}

MatrixFileLoader::~MatrixFileLoader() {
    this->file.close();
}

unsigned long MatrixFileLoader::getTotalMatrices() {
    return this->totalMatrices;
}

unsigned long MatrixFileLoader::getCurrentMatrix() {
    return this->currentMatrix;
}
