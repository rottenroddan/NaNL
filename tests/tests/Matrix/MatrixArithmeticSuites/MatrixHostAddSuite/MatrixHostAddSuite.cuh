//
// Created by steve on 7/16/2023.
//

#pragma once
#ifndef NANL_MATRIXHOSTADDSUITE_CUH
#define NANL_MATRIXHOSTADDSUITE_CUH

#include "gtest/gtest.h"
#include "../MatrixArithmeticUtility/TestMatrices.cuh"

#include <MatrixFileLoader.cuh>

class MatrixHostAddSuite : public ::testing::Test {
protected:
    static TestMatrices* smallTestMatrices;
    static TestMatrices* mediumTestMatrices;
    static TestMatrices* largeTestMatrices;

    inline static void SetUpTestSuite();

    inline static void TearDownTestSuite();
};

#include "MatrixHostAddSuite.cu"

#endif //NANL_MATRIXHOSTADDSUITE_CUH
