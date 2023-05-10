//
// Created by steve on 11/27/2022.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>

TEST(BaseMatrixTestSuite, Should_Be_Able_To_Access_Matrix_When_Index_Is_Within_Bounds) {
    NaNL::Matrix<int, NaNL::Device::Host> a(10, 33);
    NaNL::Matrix<int, NaNL::Device::Cuda> b(10, 33);

    for(unsigned int i = 0; i < a.getRows(); i++) {
        // set every i,0th index to 100.
        a[i][0] = 100;
        b[i][0] = 100;
        for(unsigned int j = 1; j < a.getCols(); j++) {
            // set every index to 10 except when J=0.
            a[i][j] = 10;
            b[i][j] = 10;
        }
    }

    for(unsigned int i = 0; i < a.getRows(); i++) {
        // check every i,0th index is 100. Host/Cuda
        ASSERT_EQ(a[i][0], 100);
        ASSERT_EQ(a.get(i, 0), 100);
        ASSERT_EQ(b[i][0], 100);
        ASSERT_EQ(b.get(i, 0), 100);
        for(unsigned int j = 1; j < a.getCols(); j++) {
            // check everything else is 10.
            ASSERT_EQ(a[i][j], 10);
            ASSERT_EQ(a.get(i, j), 10);
            ASSERT_EQ(b[i][j], 10);
            ASSERT_EQ(b.get(i, j), 10);
        }
    }
}

TEST(BaseMatrixTestSuite, Should_Return_True_When_Both_Matrices_Are_Exact_Same_Shape) {
    // Host
    NaNL::Matrix<int, NaNL::Device::Host> host_A(66, 23);
    NaNL::Matrix<int, NaNL::Device::Host> host_B(66, 23);

    ASSERT_TRUE(host_A.validateMatricesAreSameShape(host_B));

    NaNL::Matrix<int, NaNL::Device::Host> cuda_A(66, 23);
    NaNL::Matrix<int, NaNL::Device::Host> cuda_B(66, 23);

    ASSERT_TRUE(cuda_A.validateMatricesAreSameShape(cuda_B));
}

TEST(BaseMatrixTestSuite, Should_Return_False_When_Both_Matrices_Are_Not_The_Exact_Same_Shape) {
    NaNL::Matrix<int, NaNL::Device::Host> a(66, 23);
    NaNL::Matrix<int, NaNL::Device::Host> b(100, 23);
    ASSERT_TRUE(!a.validateMatricesAreSameShape(b));

    NaNL::Matrix<int, NaNL::Device::Host> c(66, 23);
    NaNL::Matrix<int, NaNL::Device::Host> d(66, 72);
    ASSERT_TRUE(!c.validateMatricesAreSameShape(d));

    NaNL::Matrix<int, NaNL::Device::Host> e(100, 10);
    NaNL::Matrix<int, NaNL::Device::Host> f(10, 100);
    ASSERT_TRUE(!e.validateMatricesAreSameShape(f));
}