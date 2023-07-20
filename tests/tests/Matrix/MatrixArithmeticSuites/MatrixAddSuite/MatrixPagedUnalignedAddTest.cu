//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include "Matrix.cuh"
#include "MatrixAddSuite.cuh"

TEST_F(MatrixAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = smallTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = smallTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

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

TEST_F(MatrixAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = mediumTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = mediumTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

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

TEST_F(MatrixAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = largeTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = largeTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b);

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

