//
// Created by steve on 6/26/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <PagedMemoryBlock.cuh>
#include <PinnedMemoryBlock.cuh>
#include <DeviceMemoryBlock.cuh>
using namespace NaNL;

TEST(MatrixDataManipulationTestSuite,  Should_Copy_Paged_To_Paged_Matrix) {
    uint64_t rows = 60;
    uint64_t cols = 40;

    NaNL::Matrix<uint64_t , PagedMemoryBlock, Unaligned> a(rows, cols);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    auto b = a.copyTo<PagedMemoryBlock, Unaligned>();

    ASSERT_EQ(a.getCols(), b.getCols());
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getTotalSize(), b.getTotalSize());
    ASSERT_EQ(a.getActualTotalSize(), b.getActualTotalSize());
    ASSERT_EQ(a.getActualCols(), b.getActualCols());
    ASSERT_EQ(a.getActualRows(), b.getActualRows());

    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            ASSERT_EQ(b[i][j], i*j);
        }
    }
}

TEST(MatrixDataManipulationTestSuite,  Should_Copy_Paged_To_Pinned_Matrix) {
    uint64_t rows = 60;
    uint64_t cols = 40;

    NaNL::Matrix<uint64_t , PagedMemoryBlock, Unaligned> a(rows, cols);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();

    ASSERT_EQ(a.getCols(), b.getCols());
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getTotalSize(), b.getTotalSize());
    ASSERT_EQ(a.getActualTotalSize(), b.getActualTotalSize());
    ASSERT_EQ(a.getActualCols(), b.getActualCols());
    ASSERT_EQ(a.getActualRows(), b.getActualRows());

    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            ASSERT_EQ(b[i][j], i*j);
        }
    }
}

TEST(MatrixDataManipulationTestSuite,  Should_Copy_Pinned_To_Paged_Matrix) {
    uint64_t rows = 60;
    uint64_t cols = 40;

    Matrix<uint64_t , PinnedMemoryBlock, Unaligned> a(rows, cols);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    auto b = a.copyTo<PagedMemoryBlock, Unaligned>();

    ASSERT_EQ(a.getCols(), b.getCols());
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getTotalSize(), b.getTotalSize());
    ASSERT_EQ(a.getActualTotalSize(), b.getActualTotalSize());
    ASSERT_EQ(a.getActualCols(), b.getActualCols());
    ASSERT_EQ(a.getActualRows(), b.getActualRows());

    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            ASSERT_EQ(b[i][j], i*j);
        }
    }
}

TEST(MatrixDataManipulationTestSuite,  Should_Copy_Pinned_To_Pinned_Matrix) {
    uint64_t rows = 60;
    uint64_t cols = 40;

    Matrix<uint64_t , PinnedMemoryBlock, Unaligned> a(rows, cols);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();

    ASSERT_EQ(a.getCols(), b.getCols());
    ASSERT_EQ(a.getRows(), b.getRows());
    ASSERT_EQ(a.getTotalSize(), b.getTotalSize());
    ASSERT_EQ(a.getActualTotalSize(), b.getActualTotalSize());
    ASSERT_EQ(a.getActualCols(), b.getActualCols());
    ASSERT_EQ(a.getActualRows(), b.getActualRows());

    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            ASSERT_EQ(b[i][j], i*j);
        }
    }
}

/**
 * Device tests must return back to host/pinned
 */
TEST(MatrixDataManipulationTestSuite,  Should_Copy_Host_To_Device_To_Host_Matrix) {
    uint64_t rows = 60;
    uint64_t cols = 40;

    Matrix<uint64_t , PinnedMemoryBlock, Unaligned> a(rows, cols);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    // copy host to device
    auto b = a.copyTo<DeviceMemoryBlock, Unaligned>();

    // copy device back to host
    auto bb = b.copyTo<PagedMemoryBlock, Unaligned>();

    ASSERT_EQ(a.getCols(), bb.getCols());
    ASSERT_EQ(a.getRows(), bb.getRows());
    ASSERT_EQ(a.getTotalSize(), bb.getTotalSize());
    ASSERT_EQ(a.getActualTotalSize(), bb.getActualTotalSize());
    ASSERT_EQ(a.getActualCols(), bb.getActualCols());
    ASSERT_EQ(a.getActualRows(), bb.getActualRows());

    for(uint64_t i = 0; i < bb.getRows(); i++) {
        for(uint64_t j = 0; j < bb.getCols(); j++) {
            ASSERT_EQ(bb[i][j], i*j);
        }
    }
}

TEST(MatrixDataManipulationTestSuite, Should_Move_Matrix_When_rValue_Is_Returned) {
    Matrix<int> a(100, 200);

    for(uint64_t i = 0; i < a.getRows(); i++) {
        for(uint64_t j = 0; j < a.getCols(); j++) {
            a[i][j] = i * j;
        }
    }

    Matrix<int> b;

    b = std::move(a);

    ASSERT_EQ(b.getRows(),100);
    ASSERT_EQ(b.getCols(),200);
    ASSERT_EQ(b.getTotalSize(), 100 * 200);
    ASSERT_EQ(b.getActualRows(), 100);
    ASSERT_EQ(b.getActualCols(), 200);
    ASSERT_EQ(b.getActualTotalSize(), 100 * 200);

    for(uint64_t i = 0; i < b.getRows(); i++) {
        for(uint64_t j = 0; j < b.getCols(); j++) {
            ASSERT_EQ(b[i][j], i*j);
        }
    }
}
