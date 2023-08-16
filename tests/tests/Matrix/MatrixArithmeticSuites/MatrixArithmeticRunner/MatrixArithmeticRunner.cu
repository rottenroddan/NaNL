//
// Created by steve on 7/16/2023.
//

#include "gtest/gtest.h"

// host operations
#include "../MatrixHostAddSuite/MatrixPagedUnalignedAddTest.cu"
#include "../MatrixHostAddSuite/MatrixPinnedUnalignedAddTest.cu"
#include "../MatrixHostAddSuite/MatrixDeviceUnalignedAddTest.cu"
#include "../MatrixHostAddSuite/MatrixPagedTensorAlignedAddTest.cu"
#include "../MatrixHostAddSuite/MatrixPinnedTensorAlignedAddTest.cu"
#include "../MatrixHostAddSuite/MatrixDeviceTensorAlignedAddTest.cu"

// cuda operations
#include "../MatrixCudaAddSuite/MatrixPagedUnalignedCudaAddTest.cu"


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}