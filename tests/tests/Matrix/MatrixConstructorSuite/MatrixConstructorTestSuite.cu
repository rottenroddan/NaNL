//
// Created by steve on 2/16/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>

TEST(MatrixConstructorTestSuite,  Should_Construct_Paged_Unaligned_Matrix_And_Store_Correct_Rows_And_Columns_When_Provided) {
    NaNL::Matrix<long, NaNL::PagedMemoryBlock, NaNL::Unaligned> a(10, 20);

    // PagedUnalligned
    ASSERT_EQ(a.getCols(), 20);
    ASSERT_EQ(a.getRows(), 10);
    ASSERT_EQ(a.getTotalSize(), 200);
}

TEST(MatrixConstructorTestSuite,  Should_Construct_Pinned_Unaligned_Matrix_And_Store_Correct_Rows_And_Columns_When_Provided) {
    NaNL::Matrix<long, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b(10, 20);

    // Cuda
    ASSERT_EQ(b.getCols(), 20);
    ASSERT_EQ(b.getRows(), 10);
    ASSERT_EQ(b.getTotalSize(), 200);
}

NaNL::Matrix<int> hostMatrixRValueHelper(unsigned long rows, unsigned long cols) {
    NaNL::Matrix<int> m(rows,cols);
    for(uint64_t i = 0; i < m.getRows(); i++) {
        for(uint64_t j = 0; j < m.getCols(); j++) {
            m[i][j] = i*j;
        }
    }
    return m;
}

NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> cudaMatrixRValueHelper(unsigned long rows, unsigned long cols) {
    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> m(rows,cols);
    for(uint64_t i = 0; i < m.getRows(); i++) {
        for(uint64_t j = 0; j < m.getCols(); j++) {
            m[i][j] = i*j;
        }
    }
    return m;
}

TEST(MatrixConstructorTestSuite, Should_Shallow_Copy_Paged_Unaligned_Matrix_When_RValue_Is_Passed_Back) {
    unsigned long rows = 10;
    unsigned long cols = 15;

    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> hostMatrix = hostMatrixRValueHelper(rows, cols);
    ASSERT_EQ(hostMatrix.getRows(), rows);
    ASSERT_EQ(hostMatrix.getCols(), cols);

    for(uint64_t i = 0; i < hostMatrix.getRows(); i++) {
        for(uint64_t j = 0; j < hostMatrix.getCols(); j++) {
            ASSERT_EQ(hostMatrix[i][j],i*j);
        }
    }
}

TEST(MatrixConstructorTestSuite, Should_Shallow_Copy_Pinned_Unaligned_Matrix_When_RValue_Is_Passed_Back) {
    unsigned long rows = 10;
    unsigned long cols = 15;

    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> cudaMatrix = cudaMatrixRValueHelper(rows, cols);
    ASSERT_EQ(cudaMatrix.getRows(), rows);
    ASSERT_EQ(cudaMatrix.getCols(), cols);

    for(uint64_t i = 0; i < cudaMatrix.getRows(); i++) {
        for(uint64_t j = 0; j < cudaMatrix.getCols(); j++) {
            ASSERT_EQ(cudaMatrix[i][j],i*j);
        }
    }
}

TEST(MatrixConstructorTestSuite, Should_Deep_Copy_Paged_Unaligned_Matrix_When_Matrix_Is_Set_To_Other) {
    unsigned long rows = 5;
    unsigned long cols = 7;

    NaNL::Matrix<double, NaNL::PagedMemoryBlock, NaNL::Unaligned> a = NaNL::Matrix<double, NaNL::PagedMemoryBlock, NaNL::Unaligned>(rows, cols);
    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = (double)(i * a.getCols() + j);
        }
    }

    // copy
    NaNL::Matrix<double, NaNL::PagedMemoryBlock, NaNL::Unaligned> b = a;

    ASSERT_EQ(b.getTotalSize(), a.getTotalSize());
    ASSERT_EQ(b.getRows(), a.getRows());
    ASSERT_EQ(b.getCols(), a.getCols());

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            ASSERT_EQ((double)(i * a.getCols() + j), b[i][j]);
        }
    }
}

TEST(MatrixConstructorTestSuite, Should_Deep_Copy_Pinned_Unaligned_Matrix_When_Matrix_Is_Set_To_Other) {
    unsigned long rows = 5;
    unsigned long cols = 7;

    NaNL::Matrix<double, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = NaNL::Matrix<double, NaNL::PinnedMemoryBlock, NaNL::Unaligned>(rows, cols);
    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = (double)(i * a.getCols() + j);
        }
    }

    // copy
    NaNL::Matrix<double, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = a;

    ASSERT_EQ(b.getTotalSize(), a.getTotalSize());
    ASSERT_EQ(b.getRows(), a.getRows());
    ASSERT_EQ(b.getCols(), a.getCols());

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            ASSERT_EQ((double)(i * a.getCols() + j), b[i][j]);
        }
    }
}