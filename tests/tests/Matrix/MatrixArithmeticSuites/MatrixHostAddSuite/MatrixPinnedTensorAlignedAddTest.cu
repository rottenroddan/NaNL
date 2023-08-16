//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include <cuda_fp16.h>
#include <Matrix.cuh>
#include "MatrixHostAddSuite.cuh"
#include <TensorCoreAligned16.cuh>

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Pinned_Tensor_Aligned_16) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception &e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Pinned_Tensor_Aligned_16) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;
        auto b = mediumTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception &e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Pinned_Tensor_Aligned_16) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;
        auto b = largeTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned16, int16_t>();;

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::TensorCoreAligned8>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception &e) {
        std::cout << e.what();
        FAIL();
    }
}