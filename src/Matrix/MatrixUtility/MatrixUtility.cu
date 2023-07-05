//
// Created by steve on 6/1/2023.
//

#pragma once
#include "MatrixUtility.cuh"

namespace NaNL {
    namespace Internal {
        template<class T>
        void MatrixUtility::addHost_Paged_Paged(const Matrix<T, PagedMemoryBlock, Unaligned> &a,
                                                                 const Matrix<T, PagedMemoryBlock, Unaligned> &b,
                                                                 Matrix<T, PagedMemoryBlock, Unaligned> &c) {
            const T *_a = a.getMatrix();
            const T *_b = b.getMatrix();
            T *_c = c.getMatrix();

            uint64_t totalThreads = NaNL::ThreadPool::getInstance()->getAllocatedThreads();

            uint64_t blockSize = a.getActualTotalSize() / totalThreads;
            uint64_t remainder = a.getActualTotalSize() - blockSize * totalThreads;
            uint64_t threadOffset = 0;

            NaNL::ThreadPool *threadPool = NaNL::ThreadPool::getInstance();
            std::deque<std::future<void>> results;

            /// create
            for (uint64_t i = 0; i < totalThreads; i++) {
                uint64_t modifiedBlockSize = (remainder == 0) ? blockSize : blockSize + 1;
                std::future<void> result = threadPool->queue([_a, _b, _c, modifiedBlockSize, threadOffset] {
                    for (uint64_t i = 0; i < modifiedBlockSize; i++) {
                        _c[threadOffset + i] = _a[threadOffset + i] + _b[threadOffset + i];
                    }
                });

                results.push_back(std::move(result));
                threadOffset += modifiedBlockSize;

                if (remainder != 0) {
                    remainder--;
                }
            }

            /// wait for threads to finish.
            for (auto &result: results) {
                result.wait();
            }
        }

        /** TODO: delete I think? */
        template<class T, template<typename> class Memory, template<class, template<typename> class> class Alignment,
                template<typename> class uMemory, template<class, template<typename> class> class uAlignment>
        void MatrixUtility::_copy(const Matrix<T, Memory, Alignment> &first, Matrix<T, uMemory, uAlignment> &second) {
            if(MemoryTypes::CudaDevice == first.getMemoryType()) {
                if(MemoryTypes::CudaPinned == second.getMemoryType() || MemoryTypes::CudaPinned == second.getMemoryType() ) {
                    T* _firstArr = first._matrix.get();
                    T* _secondArr = second._matrix.get();

                    gpuErrchk(cudaMemcpy(_secondArr, _firstArr, first.getActualTotalSize() * sizeof(T), cudaMemcpyDeviceToHost));
                } else if(MemoryTypes::CudaDevice == second.getMemoryType()) {
                    T* _fistArr = first._matrix.get();
                    T* _secondArr = second._matrix.get();
                }
            }


        }


        template<class T, template<typename> class rMemory,
                template<class, template<typename> class> class rAlignment,
                template<typename> class Memory,
                template<class, template<typename> class> class Alignment,
                template<typename> class uMemory,
                template<class, template<typename> class> class uAlignment>
        Matrix<T, rMemory, rAlignment> MatrixUtility::addHost(const Matrix<T, Memory, Alignment> &a,
                                                                    const Matrix<T, uMemory, uAlignment> &b) {
            if (!a.validateMatricesAreSameShape(b)) {
                // TODO: throw exception eventually
                //throw a.MatrixIsInvalidShape("");
            }

            // Since C is to be calculated on host, define as paged memory with return alignment.
            Matrix<T, PagedMemoryBlock, rAlignment> c(a.getRows(), a.getCols());

            // host version of these objects.
            Matrix<T, PagedMemoryBlock, Alignment> aCopyToHost;
            Matrix<T, PagedMemoryBlock, uAlignment> bCopyToHost;
            Matrix<T, PagedMemoryBlock, rAlignment> cCopyToHost;
            const Matrix<T, Memory, Alignment>* _a = nullptr;
            const Matrix<T, Memory, Alignment>* _b = nullptr;
            Matrix<T, PagedMemoryBlock, rAlignment> *_c = &c;

            // copy A to Paged memory from device.
            if(a.getMemoryType() == MemoryTypes::CudaDevice ) {
                aCopyToHost = a.template copyTo<PagedMemoryBlock, Alignment>();
                _a = &aCopyToHost;
            } else {
                _a = &a;
            }

            // copy B to Paged memory from device.
            if(b.getMemoryType() == MemoryTypes::CudaDevice ) {
                bCopyToHost = b.template copyTo<PagedMemoryBlock, uAlignment>();
                _b = &bCopyToHost;
            } else {
                _b = &b;
            }

            addHost_Paged_Paged(*_a, *_b, *_c);

            // TODO: move c to requested
            return c.template moveTo<rMemory, rAlignment>();
        }
    }
}


//template<class T>
//void NaNL::MatrixUtility::add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b) {
//
//}