//
// Created by steve on 6/1/2023.
//

#include "MatrixUtility.cuh"

template<class T>
void NaNL::MatrixUtility::_addHost_Paged_Paged(const Matrix <T, PagedMemoryBlock, Unaligned> &a,
                                               const Matrix <T, PagedMemoryBlock, Unaligned> &b,
                                               Matrix <T, PagedMemoryBlock, Unaligned> &c) {
    const T* _a = a._matrix.get();
    const T* _b = b._matrix.get();
    T* _c = c._matrix.get();

    uint64_t totalThreads = NaNL::ThreadPool::getInstance()->getAllocatedThreads();

    uint64_t blockSize = a.getActualTotalSize() / totalThreads;
    uint64_t remainder = a.getActualTotalSize() - blockSize * totalThreads;
    uint64_t threadOffset = 0;

    NaNL::ThreadPool* threadPool = NaNL::ThreadPool::getInstance();
    std::deque<std::future<void>> results;

    /// create
    for(uint64_t i = 0; i < totalThreads; i++) {
        uint64_t modifiedBlockSize = (remainder == 0) ? blockSize : blockSize + 1;
        std::future<void> result = threadPool->queue([_a,_b,_c,modifiedBlockSize,threadOffset] {
            for(uint64_t i = 0; i < modifiedBlockSize; i++) {
                _c[threadOffset + i] = _a[threadOffset + i] + _b[threadOffset + i];
            }
        });

        results.push_back(std::move(result));
        threadOffset += modifiedBlockSize;

        if(remainder != 0) {
            remainder--;
        }
    }

    /// wait for threads to finish.
    for(auto & result : results) {
        result.wait();
    }
}


template<class T, template<typename> class rMemory,
        template<class, template<typename> class> class rAlignment>
NaNL::Matrix<T, rMemory, rAlignment> NaNL::MatrixUtility::add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device) {
    if(!a.validateMatricesAreSameShape(b)) {
        // TODO: throw exception eventually
        //throw a.MatrixIsInvalidShape("");
    }

    NaNL::Matrix<T, rMemory, rAlignment> rMatrix(a.getRows(), a.getCols());

    if(MatrixDeviceOperation::Host == device) {
        _addHost_Paged_Paged<T>(a, b, rMatrix);
    } else {
        //_addCuda_Paged_Paged<T>(a,b,rMatrix);
    }

    return rMatrix;
}



//template<class T>
//void NaNL::MatrixUtility::add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b) {
//
//}