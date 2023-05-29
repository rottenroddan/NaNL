//
// Created by steve on 2/16/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>

TEST(BaseMatrixConstructorTestSuite,  Should_Construct_Matrix_And_Store_Correct_Rows_And_Columns_When_Provided) {
    NaNL::Matrix<long, NaNL::Device::Host> a(10, 20);
    NaNL::Matrix<long, NaNL::Device::Cuda> b(10, 20);

    // PagedUnalligned
    ASSERT_EQ(a.getCols(), 20);
    ASSERT_EQ(a.getRows(), 10);
    ASSERT_EQ(a.getTotalSize(), 200);

    // Cuda
    ASSERT_EQ(b.getCols(), 20);
    ASSERT_EQ(b.getRows(), 10);
    ASSERT_EQ(b.getTotalSize(), 200);
}

NaNL::Matrix<int, NaNL::Device::Host> hostMatrixRValueHelper(unsigned long rows, unsigned long cols) {
    NaNL::Matrix<int, NaNL::Device::Host> m(rows,cols);
    return m;
}

NaNL::Matrix<int, NaNL::Device::Cuda> cudaMatrixRValueHelper(unsigned long rows, unsigned long cols) {
    NaNL::Matrix<int, NaNL::Device::Cuda> m(rows,cols);
    return m;
}

TEST(BaseMatrixConstructorTestSuite, Should_Shallow_Copy_Matrix_When_RValue_Is_Passed_Back) {
    unsigned long rows = 10;
    unsigned long cols = 15;

    NaNL::Matrix<int, NaNL::Device::Host> hostMatrix = hostMatrixRValueHelper(rows, cols);
    ASSERT_EQ(hostMatrix.getRows(), rows);
    ASSERT_EQ(hostMatrix.getCols(), cols);

    NaNL::Matrix<int, NaNL::Device::Cuda> cudaMatrix = cudaMatrixRValueHelper(rows, cols);
    ASSERT_EQ(cudaMatrix.getRows(), rows);
    ASSERT_EQ(cudaMatrix.getCols(), cols);
}

TEST(BaseMatrixConstructorTestSuite, Should_Deep_Copy_Matrix_When_Matrix_Is_Set_To_Other) {
    unsigned long rows = 5;
    unsigned long cols = 7;

    NaNL::Matrix<double, NaNL::Device::Host> a = NaNL::Matrix<double, NaNL::Device::Host>(rows, cols);
    NaNL::Matrix<double, NaNL::Device::Host> b = a;

    ASSERT_EQ(b.getTotalSize(), a.getTotalSize());
    ASSERT_EQ(b.getRows(), a.getRows());
    ASSERT_EQ(b.getCols(), a.getCols());
}