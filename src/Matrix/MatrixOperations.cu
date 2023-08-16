//
// Created by steve on 7/31/2023.
//

#include "../Kernels/MatrixKernels.cuh"

namespace NaNL {
    namespace Internal {
        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class uMemory, class uAlignment,
                template<class, class> class rMemory, class rAlignment>
        void _addMatricesOnHost(const Matrix<T, Memory, Alignment> &a,
                                const Matrix<T, uMemory, uAlignment> &b,
                                Matrix<T, rMemory, rAlignment> &c) requires
        IsDerivedFromHostMemoryBlock<T, Memory, Alignment> &&
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

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class uMemory, class uAlignment,
                template<class, class> class rMemory, class rAlignment>
        void _addMatricesOnCuda(const Matrix<T, Memory, Alignment> &a,
                                const Matrix<T, uMemory, uAlignment> &b,
                                Matrix<T, rMemory, rAlignment> &c) requires
                IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment> &&
                IsDerivedFromDeviceMemoryBlock<T, uMemory, uAlignment> &&
                IsDerivedFromDeviceMemoryBlock<T, rMemory, rAlignment> {

            // fetch current cuda device number. Set device to a.
            int deviceNum;
            gpuErrchk(cudaGetDevice(&deviceNum));
            const int currentCudaDevice = deviceNum;
            cudaSetDevice(a.getCudaDevice());

            const T *_a = a.getMatrix();
            const T *_b = b.getMatrix();
            T *_c = c.getMatrix();

            dim3 threadsPerBlock(1024);
            dim3 numBlocks(std::ceil((double)a.getActualTotalSize() / (double)threadsPerBlock.x));

            Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());

            gpuErrchk(cudaDeviceSynchronize());

            // set cuda device back to current.
            gpuErrchk(cudaSetDevice(currentCudaDevice));
        }

        template<class T, template<class, class> class Memory, class Alignment,
                class R>
        Matrix<R, Memory, Alignment> _castMatricesOnCuda(const Matrix<T, Memory, Alignment> &a) //requires
            //IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment> &&
            //IsDerivedFromDeviceMemoryBlock<R, Memory, Alignment>
        {
            // fetch current cuda device number.
            int deviceNum;
            gpuErrchk(cudaGetDevice(&deviceNum));
            const int currentCudaDevice = deviceNum;

            // set cuda to 'a matrix' device.
            cudaSetDevice(a.getCudaDevice());
            auto castMatrix = Matrix<R, Memory, Alignment>(a.getRows(), a.getCols());


            dim3 threadsPerBlock(1024);
            dim3 numBlocks(std::ceil((double)a.getTotalSize() / (double)threadsPerBlock.x) );
            NaNL::Internal::Kernels::deviceMatrixCast<<<numBlocks, threadsPerBlock>>>(a.getMatrix(), castMatrix.getMatrix(), a.getActualTotalSize());

            gpuErrchk(cudaDeviceSynchronize());

            // set cuda device back to current.
            gpuErrchk(cudaSetDevice(currentCudaDevice));
            return castMatrix;
        }
    }
}