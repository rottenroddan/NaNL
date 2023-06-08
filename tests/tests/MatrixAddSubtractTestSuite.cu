//
// Created by steve on 12/18/2022.
//

//#include "gtest/gtest.h"
//#include <Matrix.cuh>
//#include "MatrixFileLoader/MatrixFileLoader.cuh"
//
//TEST(MatrixAddSubtractTestSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Host) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_100x100");
//
//    NaNL::Matrix<int, NaNL::Device::Host> a = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//    NaNL::Matrix<int, NaNL::Device::Host> b = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//    NaNL::Matrix<int, NaNL::Device::Host> truth = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//
//    a.add(b);
//
//    for(unsigned int i = 0; i < truth.getRows(); i++) {
//        for(unsigned int j = 0; j < truth.getCols(); j++) {
//            EXPECT_EQ(a[i][j], truth[i][j]);
//        }
//    }
//}
//
//TEST(MatrixAddSubtractTestSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Host) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_5000x5000");
//
//    NaNL::Matrix<int, NaNL::Device::Host> a = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//    NaNL::Matrix<int, NaNL::Device::Host> b = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//    NaNL::Matrix<int, NaNL::Device::Host> truth = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
//
//    a.add(b);
//
//    for(unsigned int i = 0; i < truth.getRows(); i++) {
//        for(unsigned int j = 0; j < truth.getCols(); j++) {
//            ASSERT_EQ(a[i][j], truth[i][j]);
//        }
//    }
//}
//
//TEST(MatrixAddSubtractTestSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Cuda) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_100x100");
//
//    NaNL::Matrix<int, NaNL::Device::Cuda> a = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//    NaNL::Matrix<int, NaNL::Device::Cuda> b = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//    NaNL::Matrix<int, NaNL::Device::Cuda> truth = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//
//    a.add(b);
//
//    for(unsigned int i = 0; i < truth.getRows(); i++) {
//        for(unsigned int j = 0; j < truth.getCols(); j++) {
//            ASSERT_EQ(a[i][j], truth[i][j]);
//        }
//    }
//}
//
//TEST(MatrixAddSubtractTestSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Cuda) {
//    MatrixFileLoader fileLoader("data/MATRIX_ADDITION_TEST_5000x5000");
//
//    NaNL::Matrix<int, NaNL::Device::Cuda> a = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//    NaNL::Matrix<int, NaNL::Device::Cuda> b = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//    NaNL::Matrix<int, NaNL::Device::Cuda> truth = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Cuda>();
//
//    a.add(b);
//
//    for(unsigned int i = 0; i < truth.getRows(); i++) {
//        for(unsigned int j = 0; j < truth.getCols(); j++) {
//            ASSERT_EQ(a[i][j], truth[i][j]);
//        }
//    }
//}