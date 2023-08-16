//
// Created by steve on 8/3/2023.
//

//
// Created by steve on 7/16/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixHostAddSuite.cuh"
#include <TensorCoreAligned32.cuh>

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}