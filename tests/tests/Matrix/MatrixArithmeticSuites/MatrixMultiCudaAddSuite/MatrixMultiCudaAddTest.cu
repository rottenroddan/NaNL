//
// Created by steve on 9/4/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixMultiCudaAddSuite.cuh"

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixMultiCudaAddSuite.cuh"

TEST_F(MatrixMultiCudaAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Cuda_Multi_Device) {
    try {
        // create all these on cuda device one.
        cudaSetDevice(0);
        auto a = smallTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        std::vector<std::thread> threads;

        threads.emplace_back([&] {
            // cuda initializer
            cudaSetDevice(1);
            auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda).copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();


            ASSERT_EQ(c.getRows(), truth.getRows());
            ASSERT_EQ(c.getCols(), truth.getCols());

            for (unsigned int i = 0; i < c.getRows(); i++) {
                for (unsigned int j = 0; j < c.getCols(); j++) {
                    ASSERT_EQ(c[i][j], truth[i][j]);
                }
            }
        });

        for(auto& thread : threads) {
            thread.join();
        }

    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixMultiCudaAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        // create all these on cuda device one.
        cudaSetDevice(0);
        auto a = mediumTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        std::vector<std::thread> threads;

        threads.emplace_back([&] {
            // cuda initializer
            cudaSetDevice(1);
            auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda).copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();


            ASSERT_EQ(c.getRows(), truth.getRows());
            ASSERT_EQ(c.getCols(), truth.getCols());

            for (unsigned int i = 0; i < c.getRows(); i++) {
                for (unsigned int j = 0; j < c.getCols(); j++) {
                    ASSERT_EQ(c[i][j], truth[i][j]);
                }
            }
        });

        for(auto& thread : threads) {
            thread.join();
        }

    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixMultiCudaAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        // create all these on cuda device one.
        cudaSetDevice(0);
        auto a = largeTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        std::vector<std::thread> threads;

        threads.emplace_back([&] {
            // cuda initializer
            cudaSetDevice(1);
            auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda).copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();


            ASSERT_EQ(c.getRows(), truth.getRows());
            ASSERT_EQ(c.getCols(), truth.getCols());

            for (unsigned int i = 0; i < c.getRows(); i++) {
                for (unsigned int j = 0; j < c.getCols(); j++) {
                    ASSERT_EQ(c[i][j], truth[i][j]);
                }
            }
        });

        for(auto& thread : threads) {
            thread.join();
        }

    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}