//
// Created by steve on 2/26/2023.
//

#include <iostream>
#include <PinnedMemoryBlock.cuh>
#include <PagedMemoryBlock.cuh>
#include <DeviceMemoryBlock.cuh>
#include <Matrix.cuh>
#include <MatrixFileLoader.cuh>
#include <MatrixOutBinaryFileLoader.cuh>
#include <MatrixInBinaryFileLoader.cuh>
#include <TensorCoreAligned32.cuh>
#include <type_traits>

#include <math.h>

float p2p_copy (size_t size)
{
    int *pointers[2];

    cudaSetDevice (0);
    cudaDeviceEnablePeerAccess (1, 0);
    cudaMalloc (&pointers[0], size);

    cudaSetDevice (1);
    cudaDeviceEnablePeerAccess (0, 0);
    cudaMalloc (&pointers[1], size);

    cudaEvent_t begin, end;
    cudaEventCreate (&begin);
    cudaEventCreate (&end);

    cudaEventRecord (begin);
    cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord (end);
    cudaEventSynchronize (end);

    float elapsed;
    cudaEventElapsedTime (&elapsed, begin, end);
    elapsed /= 1000;

    cudaSetDevice (0);
    cudaFree (pointers[0]);

    cudaSetDevice (1);
    cudaFree (pointers[1]);

    cudaEventDestroy (end);
    cudaEventDestroy (begin);

    return elapsed;
}

void test() {
    std::vector<std::thread> threads;

    cudaSetDevice(0);
    NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> A(10000,10000);
    auto x = A.copyTo<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();



    p2p_copy(4000);

    threads.emplace_back([&] {
        // cuda initializer
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> m(10000, 10000);

        DWORD threadId = GetCurrentThreadId();
        NaNL::Logger* logger = NaNL::Logger::GetInstance();
        logger->begin(threadId, "3080-Ti", "");

        for(uint64_t i = 0; i < 100; i++) {
            m.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(A, NaNL::MatrixAddOperation::Cuda);
        }

        logger->end(threadId);
        logger->log(threadId);
    });

    threads.emplace_back([&] {
        // cuda initializer
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> m(10000, 10000);

        DWORD threadId = GetCurrentThreadId();
        NaNL::Logger* logger = NaNL::Logger::GetInstance();
        logger->begin(threadId, "2080-Ti", "");

        for(uint64_t i = 0; i < 100; i++) {
            m.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(A, NaNL::MatrixAddOperation::Cuda);
        }

        logger->end(threadId);
        logger->log(threadId);
    });


    for(auto& thread : threads) {
        thread.join();
    }
}

void __global__ test(unsigned long a, unsigned long b, unsigned long c, unsigned long d, unsigned long e, unsigned long f, unsigned long g, unsigned long h) {
    printf("%u %u %u %u %u %u %u %u \n", a, b, c, d, e, f, g, h);
}


int main() {
    //test();

//    dim3 totalThreads(1024);
//    dim3 numblocks(1000);
//
//    test<<<numblocks, totalThreads>>>(10, 10, 10, 10, 10, 10, 10, 10);
//
//    cudaDeviceSynchronize();

    NaNL::Matrix<uint32_t, NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32> u(2,2);
    u[0][0] = 5;
    u[0][1] = 9;
    u[1][0] = 4;
    u[1][1] = 3;

    auto c = u.add<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>(u, NaNL::MatrixAddOperation::Host);

    for(uint64_t i = 0; i < c.getRows(); i++) {
        for(uint64_t j = 0; j < c.getCols(); j++) {
            std::cout << c[i][j] << std::endl;
        }
    }
    DWORD threadId = GetCurrentThreadId();
    NaNL::Logger* logger = NaNL::Logger::GetInstance();


    using type = int;
    NaNL::Matrix<type, NaNL::PagedMemoryBlock, NaNL::Unaligned> a(24000, 24000);
    NaNL::Matrix<type, NaNL::PagedMemoryBlock, NaNL::Unaligned> b(24000, 24000);


    logger->begin(threadId, "AVX", "");
    for(int i = 0; i < 25; i++) {
        a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Host);
    }
    logger->end(threadId);
    logger->log(threadId);

    logger->begin(threadId, "Non-AVX", "");
    for(int i = 0; i < 25; i++) {
        a.add<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixAddOperation::Host);
    }
    logger->end(threadId);
    logger->log(threadId);


    logger->begin(threadId, "Cuda No alignment difference", "");

    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32> aa(24000, 24000);
    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32> bb(24000, 24000);

    auto d = aa.add<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>(bb, NaNL::MatrixAddOperation::Cuda);

    logger->end(threadId);
    logger->log(threadId);


    logger->begin(threadId, "Cuda Alignment difference", "");

    auto e = aa.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(bb, NaNL::MatrixAddOperation::Cuda);

    logger->end(threadId);
    logger->log(threadId);

    logger->begin(threadId, "Host No Alignment difference", "");

    auto g = aa.add<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>(bb, NaNL::MatrixAddOperation::Host);

    logger->end(threadId);
    logger->log(threadId);

    logger->begin(threadId, "Host Alignment difference", "");

    auto f = aa.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(bb, NaNL::MatrixAddOperation::Host);

    logger->end(threadId);
    logger->log(threadId);

    return 0;
}