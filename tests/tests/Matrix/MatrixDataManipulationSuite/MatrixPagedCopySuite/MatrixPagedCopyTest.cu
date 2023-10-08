//
// Created by steve on 6/26/2023.
//

#include "MatrixPagedCopySuite.cuh"

using namespace NaNL;

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int64_t>(i,j);
            auto b = a.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Pinned_Unaligned_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t>(i,j);
            auto d = a.copyTo<PinnedMemoryBlock, Unaligned>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Pinned_TensorAligned8_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<char>(i,j);
            auto d = a.copyTo<PinnedMemoryBlock, TensorCoreAligned8>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Pinned_TensorAligned16_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<uint16_t >(i,j);
            auto d = a.copyTo<PinnedMemoryBlock, TensorCoreAligned16>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Pinned_TensorAligned32_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<uint32_t >(i,j);
            auto d = a.copyTo<PinnedMemoryBlock, TensorCoreAligned32>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Device_Unaligned_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t>(i,j);
            auto d = a.copyTo<DeviceMemoryBlock, Unaligned>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Device_Tensor8_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<float>(i,j);
            auto d = a.copyTo<DeviceMemoryBlock, TensorCoreAligned8>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Device_Tensor16_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<float>(i,j);
            auto d = a.copyTo<DeviceMemoryBlock, TensorCoreAligned16>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPagedCopySuite,  Should_Copy_Paged_Unaligned_To_Device_Tensor32_Back_To_Paged_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<float>(i,j);
            auto d = a.copyTo<DeviceMemoryBlock, TensorCoreAligned32>();
            auto b = d.copyTo<PagedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}
