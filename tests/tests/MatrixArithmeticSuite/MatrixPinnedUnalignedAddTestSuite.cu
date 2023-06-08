//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include "MatrixFileLoader/MatrixFileLoader.cuh"

//TEST(MatrixAddSubtractTestSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Host) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_100x100");
//
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//
//    auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b);
//
//    ASSERT_EQ(c.getRows(), truth.getRows());
//    ASSERT_EQ(c.getCols(), truth.getCols());
//
//    for (unsigned int i = 0; i < c.getRows(); i++) {
//        for (unsigned int j = 0; j < c.getCols(); j++) {
//            ASSERT_EQ(c[i][j], truth[i][j]);
//        }
//    }
//}
//
//TEST(MatrixAddSubtractTestSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Host) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_5000x5000");
//
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = fileLoader.readValuesIntoMatrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//
//    auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b);
//
//    ASSERT_EQ(c.getRows(), truth.getRows());
//    ASSERT_EQ(c.getCols(), truth.getCols());
//
//    for (unsigned int i = 0; i < c.getRows(); i++) {
//        for (unsigned int j = 0; j < c.getCols(); j++) {
//            ASSERT_EQ(c[i][j], truth[i][j]);
//        }
//    }
//}