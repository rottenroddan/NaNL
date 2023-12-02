//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include "MatrixHostAddSuite.cuh"

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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