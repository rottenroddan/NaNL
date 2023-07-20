//
// Created by steve on 7/16/2023.
//

#include "TestMatrices.cuh"

TestMatrices::TestMatrices(std::string filePath) {
    MatrixFileLoader fileLoader(filePath);

    // read file into Matrices.
    a = fileLoader.readValuesIntoMatrix<int>();
    b = fileLoader.readValuesIntoMatrix<int>();
    truth = fileLoader.readValuesIntoMatrix<int>();
}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfA() {
    return a.template copyTo<rMemory, rAlignment>();
}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfB() {
    return b.template copyTo<rMemory, rAlignment>();
}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfTruth() {
    return truth.template copyTo<rMemory, rAlignment>();
}