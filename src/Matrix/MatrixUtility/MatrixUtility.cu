//
// Created by steve on 6/1/2023.
//

#pragma once
#include "MatrixUtility.cuh"
#include <Windows.h>

namespace NaNL {
    namespace Internal {

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class uMemory, class uAlignment,
                template<class, class> class rMemory, class rAlignment>
        void MatrixUtility::addHost_Paged_Paged(const Matrix<T, Memory, Alignment> &a,
                                                const Matrix<T, uMemory, uAlignment> &b,
                                                Matrix<T, rMemory, rAlignment> &c)
                                                requires IsDerivedFromHostMemoryBlock<T, Memory, Alignment> &&
                                                        IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment> &&
                                                        IsDerivedFromHostMemoryBlock<T, rMemory, rAlignment> {
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
        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class uMemory, class uAlignment>
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


        template<class T, template<class, class> class rMemory,
                class rAlignment,
                template<class, class> class Memory,
                class Alignment,
                template<class, class> class uMemory,
                class uAlignment>
        Matrix<T, rMemory, rAlignment> MatrixUtility::addHost(const Matrix<T, Memory, Alignment> &a,
                                                                    const Matrix<T, uMemory, uAlignment> &b) {
            if (!a.validateMatricesAreSameShape(b)) {
                // TODO: throw exception eventually
                //throw a.MatrixIsInvalidShape("");
            }

            // Since C is to be calculated on host, define as paged memory with return alignment.
            Matrix<T, PagedMemoryBlock, rAlignment> c(a.getRows(), a.getCols());

            // a & b are both derived from HostMemoryBlock
            if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                         && IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
                addHost_Paged_Paged(a, b, c);
            }

            // Only a is derived from HostMemoryBlock
            else if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                        && !IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
                addHost_Paged_Paged(a, b.template copyTo<uMemory, uAlignment>(), c);
            }

            // Only b is derived from HostMemoryBlock
            else if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                              && !IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
                addHost_Paged_Paged(a.template copyTo<uMemory, uAlignment>(), b, c);
            }

            // Neither a or b derived from HostMemoryBlock
            else {
                addHost_Paged_Paged(a.template copyTo<uMemory, uAlignment>(), b.template copyTo<uMemory, uAlignment>(), c);
            }

            // TODO: move c to requested
            return c.template moveTo<rMemory, rAlignment>();
        }
    }
}


//template<class T>
//void NaNL::MatrixUtility::add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b) {
//
//}