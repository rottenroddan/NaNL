//
// Created by steve on 2/26/2023.
//
#include <chrono>
#include <iostream>
#include <memory>
#include <DeviceMemoryBlock.cuh>
#include <PagedMemoryBlock.cuh>
#include <PinnedMemoryBlock.cuh>
#include <../Alignment/Unaligned.cuh>
#include <Matrix.cuh>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>

//template<class T, template<typename> class Memory = NaNL::PagedMemoryBlock,
//        template<class, template<typename > class >class Alignment = NaNL::Unaligned>
//class BaseMatrix : protected Alignment<T,Memory> {
//public:
//    static_assert(std::is_base_of<NaNL::BaseAlignment<T, Memory> ,Alignment<T, Memory>>::value, "Template argument 'Alignment' must inherit from BaseAlignment class." );
//
//    BaseMatrix(uint64_t rows, uint64_t cols) : Alignment<T, Memory>(rows, cols)  {
//
//    }
//
//};

using namespace NaNL;

class A {
public:
    int x;

    explicit A(int val) {
        x = val;
    }

    A(const A &a) {
        this->x = a.x;
    }

    A& operator=(const A &copyA) noexcept {
        this->x = 99;
        return *this;
    }

    A& operator=(A &&moveA)  noexcept {
        this->x = moveA.x;
        return *this;
    }
};

int main() {

//    A a(10);
//    A b(5);
//
//    b = std::move(a);
//    std::cout << b.x << std::endl;

    NaNL::Matrix<int, PagedMemoryBlock, Unaligned> a(100, 100);
    NaNL::Matrix<int, PagedMemoryBlock, Unaligned> b;

    a.add<PagedMemoryBlock, Unaligned>(b);

    b = std::move(a);


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