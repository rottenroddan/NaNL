//
// Created by steve on 7/16/2023.
//

#include "gtest/gtest.h"

#include "../MatrixAddSuite/MatrixPagedUnalignedAddTest.cu"
#include "../MatrixAddSuite/MatrixPinnedUnalignedAddTest.cu"
#include "../MatrixAddSuite/MatrixDeviceUnalignedAddTest.cu"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}