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

            uint64_t desiredThreads = NaNL::ThreadPool::getInstance()->getAllocatedThreads();
            uint64_t totalThreads = a.getRows() >= desiredThreads ? desiredThreads : a.getRows();
            uint64_t rowsPerThread = a.getRows() / totalThreads;    // rounds down.
            uint64_t remainder = a.getRows() % totalThreads;
            uint64_t maxRowForThread = rowsPerThread + (remainder != 0 ? 1 : 0);

            NaNL::ThreadPool *threadPool = NaNL::ThreadPool::getInstance();
            std::deque<std::future<void>> results;

            uint64_t rowOffset = 0;
            for (uint64_t i = 0; i < totalThreads; i++) {
                std::future<void> result = threadPool->queue([&a, &b, &c, rowOffset, maxRowForThread] {
                    for(uint64_t i = rowOffset; i < maxRowForThread; i++) {
                        for(uint64_t j = 0; j < a.getCols(); j++) {
                            c[i][j] = a[i][j] + b[i][j];
                        }
                    }
                });
                results.push_back(std::move(result));


                rowOffset += rowsPerThread + (remainder != 0 ? 1 : 0);
                if (remainder != 0) {
                    remainder--;
                }
                maxRowForThread += rowsPerThread + (remainder != 0 ? 1 : 0);
            }

            /// wait for threads to finish.
            for (auto &result: results) {
                result.wait();
            }

//            std::cout << "A: " << a[1326][1088] << std::endl;
//            std::cout << "B: " << b[1326][1088] << std::endl;
//            std::cout << "C: " << c[1326][1088] << std::endl;
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
//            int deviceNum;
//            gpuErrchk(cudaGetDevice(&deviceNum));
//            const int currentCudaDevice = deviceNum;
//            cudaSetDevice(a.getCudaDevice());

            int currentDevice;
            cudaGetDevice(&currentDevice);

            const T *_a = a.getMatrix();
            const T *_b = b.getMatrix();
            T *_c = c.getMatrix();

            dim3 threadsPerBlock(1024);
            dim3 numBlocks(std::ceil((double)a.getActualTotalSize() / (double)threadsPerBlock.x));

            //
            if(a.getCudaDevice() != currentDevice && b.getCudaDevice() != currentDevice) {
                auto aa = a.template copyTo<Memory, Alignment>();
                auto bb = b.template copyTo<Memory, Alignment>();
                _a = aa.getMatrix();
                _b = bb.getMatrix();
                Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                gpuErrchk(cudaDeviceSynchronize());
            } else if(a.getCudaDevice() != currentDevice) {
                auto aa = a.template copyTo<Memory, Alignment>();
                _a = aa.getMatrix();
                Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                gpuErrchk(cudaDeviceSynchronize());
            } else if(b.getCudaDevice() != currentDevice) {
                auto bb = b.template copyTo<Memory, Alignment>();
                _b = bb.getMatrix();
                Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                gpuErrchk(cudaDeviceSynchronize());
            } else {
                Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }

        template<class T, template<class, class> class Memory, class Alignment,
                class R>
        static Matrix<R, Memory, Alignment> _castMatricesOnCuda(const Matrix<T, Memory, Alignment> &a) //requires
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
            dim3 numBlocks(std::ceil((double)a.getActualTotalSize() / (double)threadsPerBlock.x) );
            NaNL::Internal::Kernels::deviceMatrixCast<<<numBlocks, threadsPerBlock>>>(a.getMatrix(), castMatrix.getMatrix(), a.getActualTotalSize());

            gpuErrchk(cudaDeviceSynchronize());

            // set cuda device back to current.
            gpuErrchk(cudaSetDevice(currentCudaDevice));
            return castMatrix;
        }

        template<class T>
        static void testAgain(T a) {

        }

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class rMemory, class rAlignment>
        static void _copyDeviceToDeviceStride(Matrix<T, Memory, Alignment> &src, Matrix<T, rMemory, rAlignment> &dst) {
            std::vector<cudaStream_t> streams;
            streams.reserve(src.getRows());

            // async device to device memcpy.
            for (uint64_t i = 0; i < dst.getRows(); i++) {
                streams.emplace_back();
                gpuErrchk(cudaMemcpyAsync(dst.getMatrix() + dst.getActualCols() * i,
                                          src.getMatrix() + src.getActualCols() * i,
                                          src.getActualCols() * sizeof(T),
                                          cudaMemcpyDeviceToDevice, streams[i]));
            }

            // sync and cleanup
            for(cudaStream_t stream: streams) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class rMemory, class rAlignment>
        static void _copyDeviceToDeviceContinous(Matrix<T, Memory, Alignment> &src, Matrix<T, rMemory, rAlignment> &dst) {
            gpuErrchk(cudaMemcpy(dst.getMatrix(), src.getMatrix(), src.getActualTotalSize() * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class rMemory, class rAlignment>
        static void _copyPeerToPeerContinous(Matrix<T, Memory, Alignment> &src, Matrix<T, rMemory, rAlignment> &dst) {
            int srcDevice = src.getCudaDevice();
            int dstDevice = dst.getCudaDevice();
            gpuErrchk(cudaMemcpyPeer(dst.getMatrix(), dstDevice, src.getMatrix(), srcDevice, src.getActualTotalSize() * sizeof(T)));
        }

        template<class T, template<class, class> class Memory, class Alignment,
                template<class, class> class rMemory, class rAlignment, class R>
        static Matrix<R, rMemory, rAlignment> _copyDeviceToDevice(Matrix<T, Memory, Alignment> &src) {
            Matrix<R, rMemory, rAlignment> copyMatrix(src.getRows(), src.getCols());
            bool isSameDevice = copyMatrix.getCudaDevice() == src.getCudaDevice() ? true : false;

            // if actual size of both matrices are the same. 1:1 copy.
            // else copy each row in respect of their alignment.
            if (src.getActualRows() == copyMatrix.getActualRows()
                && src.getActualCols() == copyMatrix.getActualCols()) {
                if constexpr (std::is_same_v<T, R>) {
                    if(isSameDevice) {
                        Internal::_copyDeviceToDeviceContinous(src, copyMatrix);
                    } else {
                        _copyPeerToPeerContinous(src, copyMatrix);
                    }
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Alignment, R>(src);
                    if(isSameDevice) {
                        _copyDeviceToDeviceContinous(castMatrix, copyMatrix);
                    } else {
                        _copyPeerToPeerContinous(castMatrix, copyMatrix);
                    }
                }
            } else {
                if constexpr (std::is_same_v<T,R>) {
                    if(isSameDevice) {
                        _copyDeviceToDeviceStride(src, copyMatrix);
                    } else {

                    }
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Alignment, R>(src);
                    if(isSameDevice) {
                        _copyDeviceToDeviceStride(castMatrix, copyMatrix);
                    } else {

                    }
                }
            }

            return copyMatrix;
        }
    }
}