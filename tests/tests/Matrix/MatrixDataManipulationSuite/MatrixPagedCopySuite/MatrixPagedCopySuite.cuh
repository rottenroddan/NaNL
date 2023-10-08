//
// Created by steve on 9/27/2023.
//

#ifndef NANL_MATRIXPAGEDCOPYSUITE_CUH
#define NANL_MATRIXPAGEDCOPYSUITE_CUH

#include <gtest/gtest.h>
#include <Matrix.cuh>
#include <PagedMemoryBlock.cuh>
#include <PinnedMemoryBlock.cuh>
#include <DeviceMemoryBlock.cuh>
#include <TensorCoreAligned8.cuh>
#include <TensorCoreAligned16.cuh>
#include <TensorCoreAligned32.cuh>

#include "../Util/MatrixDataManipulationUtil.cuh"


class MatrixPagedCopySuite : public ::testing::Test {

protected:
    const uint64_t ROWS = 100;
    const uint64_t COLS = 100;

    static void SetUpTestSuite();

    static void TearDownTestSuite();
};

#endif //NANL_MATRIXPAGEDCOPYSUITE_CUH
