//
// Created by steve on 8/11/2023.
//

#pragma once
#ifndef NANL_MATRIXCUDAADDSUITE_CUH
#define NANL_MATRIXCUDAADDSUITE_CUH

#include "gtest/gtest.h"
#include "../MatrixArithmeticUtility/TestMatrices.cuh"

#include <MatrixFileLoader.cuh>

class MatrixCudaAddSuite : public ::testing::Test {
protected:
    static TestMatrices* smallTestMatrices;
    static TestMatrices* mediumTestMatrices;
    static TestMatrices* largeTestMatrices;

    inline static void SetUpTestSuite();

    inline static void TearDownTestSuite();
};

#include "MatrixCudaAddSuite.cu"

#endif //NANL_MATRIXHOSTADDSUITE_CUH
