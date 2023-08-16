//
// Created by steve on 7/16/2023.
//

#pragma once
#ifndef NANL_TESTMATRICES_CUH
#define NANL_TESTMATRICES_CUH

#include <string>
#include <Matrix.cuh>
#include <MatrixFileLoader.cuh>

class TestMatrices {
    MatrixFileLoader fileLoader;
    NaNL::Matrix<int> a;
    NaNL::Matrix<int> b;
    NaNL::Matrix<int> truth;
public:
    inline explicit TestMatrices(std::string filePath);

    template<template<class, class> class rMemory, class rAlignment>
    inline NaNL::Matrix<int, rMemory, rAlignment> getCopyOfA();

    template<template<class, class> class rMemory, class rAlignment>
    inline NaNL::Matrix<int, rMemory, rAlignment> getCopyOfB();

    template<template<class, class> class rMemory, class rAlignment>
    inline NaNL::Matrix<int, rMemory, rAlignment> getCopyOfTruth();

};

#include "TestMatrices.cu"

#endif //NANL_TESTMATRICES_CUH
