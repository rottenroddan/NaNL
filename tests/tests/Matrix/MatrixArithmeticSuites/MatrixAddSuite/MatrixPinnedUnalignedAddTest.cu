//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include "MatrixAddSuite.cuh"

TEST_F(MatrixAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = smallTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = smallTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = smallTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = mediumTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = mediumTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = mediumTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = largeTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = largeTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = largeTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}