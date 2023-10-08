//
// Created by steve on 9/28/2023.
//

#ifndef NANL_MATRIXPINNEDCOPYSUITE_CUH
#define NANL_MATRIXPINNEDCOPYSUITE_CUH

#include <gtest/gtest.h>
#include <iostream>

#include "../Util/MatrixDataManipulationUtil.cuh"

const uint64_t ROWS = 100;
const uint64_t COLS = 100;

class MatrixPinnedCopySuite : public ::testing::Test {
protected:
    static void SetUpTestSuite();

    static void TearDownTestSuite();

    void test();
};


#endif //NANL_MATRIXPINNEDCOPYSUITE_CUH
