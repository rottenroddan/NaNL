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

int main() {
    //test();




    double k = 200.0;
    std::cout << sin(k) << std::endl;

    DWORD threadId = GetCurrentThreadId();
    NaNL::Logger* logger = NaNL::Logger::GetInstance();
    logger->begin(threadId, "File Loader test", "");

    NaNL::Matrix<uint64_t, NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32> u(2,2);
    u[0][0] = 5;
    u[0][1] = 9;
    u[1][0] = 4;
    u[1][1] = 3;

    auto c = u.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(u);

    for(uint64_t i = 0; i < c.getRows(); i++) {
        for(uint64_t j = 0; j < c.getCols(); j++) {
            std::cout << c[i][j] << std::endl;
        }
    }




   // test(aa);
    //test(bb);

    //static_assert(is_base<Base<int>, A<int>>::value, "If error: A is not derived from Base");

//    A<int> aa;
//    test(aa);

    //B bb;
    //test(bb);

//    A a(10);
//    A b(5);
//
//    b = std::move(a);
//    std::cout << b.x << std::endl;

   // std::cout << sizeof(size_t) << std::endl;

//    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32> x(100, 100);
//
//    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> y(100, 100);
//
//    std::cout << y[0][100] << std::endl;


//    NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b(100, 100);
//    NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> d(100, 100);
//
//    //NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned>::addHost<>(b, d);
//
//    NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> a(100,100);
//    NaNL::Matrix<float, NaNL::PagedMemoryBlock, NaNL::Unaligned> aa(100,100);
//    NaNL::Matrix<float, NaNL::PagedMemoryBlock, NaNL::Unaligned> aaa(100,100);
//
//    NaNL::Matrix<double, NaNL::PagedMemoryBlock, NaNL::Unaligned> jjj(200,200);
//
//
//    b.getMatrix();
//
//    std::cout << a[0][0] << std::endl;

   // matrixTest(a);
   // matrixTest(b);
   // matrixTest(d);
    //matrixTest(d);

    //a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b);


//    std::chrono::time_point<std::chrono::high_resolution_clock> pagedStart, pagedEnd, pinnedStart, pinnedEnd;
//    std::chrono::time_point<std::chrono::high_resolution_clock> pagedAllocStart, pagedAllocEnd, pinnedAllocStart, pinnedAllocEnd, pinnedAlloc2Start, pinnedAlloc2End;
//
//    uint64_t cols = 4000000;
//
//    pagedAllocStart = std::chrono::high_resolution_clock::now();
//    int64_t* _pagedArr = new int64_t[cols];
//    pagedAllocEnd = std::chrono::high_resolution_clock::now();
//
//    int64_t* _pinnedArr = nullptr;
//    int64_t* _pinnedArrTemp = nullptr;
//
//    pinnedAllocStart = std::chrono::high_resolution_clock::now();
//    cudaMallocHost((void**)&_pinnedArr, cols*sizeof(uint64_t));
//    pinnedAllocEnd = std::chrono::high_resolution_clock::now();
//
//    pinnedAlloc2Start = std::chrono::high_resolution_clock::now();
//    cudaMallocHost((void**)&_pinnedArrTemp, cols*sizeof(uint64_t));
//    pinnedAlloc2End = std::chrono::high_resolution_clock::now();
//
//    pagedStart = std::chrono::high_resolution_clock::now();
//    for(uint64_t i = 0; i < cols; i++) {
//        _pagedArr[i] = i*i;
//    }
//    pagedEnd = std::chrono::high_resolution_clock::now();
//
//
//
//    pinnedStart = std::chrono::high_resolution_clock::now();
//    for(uint64_t i = 0; i < cols; i++) {
//        _pinnedArr[i] = i*i;
//    }
//    pinnedEnd = std::chrono::high_resolution_clock::now();
//
//    std::cout << "Paged Alloc time:      " << (pagedAllocEnd - pagedAllocStart).count() << "ms." << std::endl;
//    std::cout << "Pinned Alloc time:     " << (pinnedAllocEnd - pinnedAllocStart).count() << "ms." << std::endl;
//    std::cout << "2nd Pinned Alloc time: " << (pinnedAlloc2End - pinnedAlloc2Start).count() << "ms." << std::endl;
//    std::cout << "Paged time:  " << (pagedEnd - pagedStart).count() << "ms." << std::endl;
//    std::cout << "Pinned time: " << (pinnedEnd - pinnedStart).count() << "ms." << std::endl;

//    uint64_t rows = 10;
//    uint64_t cols = 10;
//
//    NaNL::Matrix<int> p(rows, cols);
//
//    for(int i = 0; i < rows; i++) {
//        for(int j = 0; j < 10; j++) {
//            p[i][j] = i * j;
//        }
//    }
//
//    int *hostPtr;
//    int *devicePtr;
//
//    gpuErrchk(cudaSetDevice(0));
//    gpuErrchk(cudaMalloc(&devicePtr, sizeof(int)));
//
//    cudaPointerAttributes attribute;
//    cudaPointerGetAttributes(&attribute, devicePtr);
//
//    if(attribute.type == cudaMemoryType::cudaMemoryTypeHost) {
//
//    }
//
//    auto copyM = p.copyTo<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
//
//    auto x = p.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(p, NaNL::MatrixDeviceOperation::Host);
//
//
//    std::cout << copyM.getTotalSize() << std::endl;
//    std::cout << x.getTotalSize() << std::endl;
//
//    for(int i = 0; i < rows; i++) {
//        for(int j = 0; j < cols; j++) {
//            std::cout << copyM[i][j] << std::endl;
//        }
//    }

    //NaNL::Unaligned<int, NaNL::PagedMemoryBlock> un(100, 100);

    //BaseMatrix<int> a(100 , 100);

//    cudaFree(_pinnedArr);
//    cudaFree(_pinnedArrTemp);

    return 0;
}