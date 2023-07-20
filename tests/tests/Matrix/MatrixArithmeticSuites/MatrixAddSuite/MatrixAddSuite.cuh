//
// Created by steve on 7/16/2023.
//

#pragma once
#ifndef NANL_MATRIXADDSUITE_CUH
#define NANL_MATRIXADDSUITE_CUH

#include "gtest/gtest.h"
#include "../MatrixArithmeticUtility/TestMatrices.cuh"

#include <MatrixFileLoader.cuh>

class MatrixAddSuite : public ::testing::Test {
protected:
    static TestMatrices* smallTestMatrices;
    static TestMatrices* mediumTestMatrices;
    static TestMatrices* largeTestMatrices;

    inline static void SetUpTestSuite();

    inline static void TearDownTestSuite();
};

#include "MatrixAddSuite.cu"

#endif //NANL_MATRIXADDSUITE_CUH
