//
// Created by steve on 9/28/2023.
//

#include "MatrixPinnedCopySuite.cuh"

using namespace NaNL;

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int64_t, PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Pinned_TensorAligned8) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int8_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, TensorCoreAligned8>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Pinned_TensorAligned16) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int16_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, TensorCoreAligned16>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Pinned_TensorAligned32) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, TensorCoreAligned32>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_TensorAligned32_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t , PinnedMemoryBlock, TensorCoreAligned32>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_TensorAligned16_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int16_t , PinnedMemoryBlock, TensorCoreAligned32>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_TensorAligned8_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int8_t , PinnedMemoryBlock, TensorCoreAligned32>(i,j);
            auto b = a.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(b);
        }
    }
}


TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Paged_Unaligned_Back_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int64_t, PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<PagedMemoryBlock, Unaligned>();
            auto c = b.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(c);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Device_Unaligned_Back_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int64_t, PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<DeviceMemoryBlock, Unaligned>();
            auto c = b.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(c);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Device_TensorAligned32_Back_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<DeviceMemoryBlock, TensorCoreAligned32>();
            auto c = b.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(c);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Device_TensorAligned16_Back_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<DeviceMemoryBlock, TensorCoreAligned16>();
            auto c = b.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(c);
        }
    }
}

TEST_F(MatrixPinnedCopySuite, Should_Copy_Pinned_Unaligned_To_Device_TensorAligned8_Back_To_Pinned_Unaligned) {
    for(uint64_t i = 1; i < ROWS; i++) {
        for(uint64_t j = 1; j < COLS; j++) {
            auto a = preload_test_matrix<int32_t , PinnedMemoryBlock, Unaligned>(i,j);
            auto b = a.copyTo<DeviceMemoryBlock, TensorCoreAligned8>();
            auto c = b.copyTo<PinnedMemoryBlock, Unaligned>();
            validate_test_matrix(c);
        }
    }
}