//
// Created by steve on 8/3/2023.
//

#include <Matrix.cuh>
#include <TensorCoreAligned8.cuh>

#include "gtest/gtest.h"
#include "MatrixHostAddSuite.cuh"

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Host_Tensor8_Aligned) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Host_Tensor8_Aligned) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Host_Tensor8_Aligned) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>().copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8, int8_t>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned8>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);

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