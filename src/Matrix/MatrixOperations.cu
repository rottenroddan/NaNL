//
// Created by steve on 7/31/2023.
//

#include "../Kernels/MatrixKernels.cuh"
#include <any>

namespace NaNL {
    namespace Internal {
        template<typename... Matrices>
        bool compareActualMatrixSize(Matrices... matrices) {
            bool isAllMatricesEqual = true;
            uint64_t rows, cols, totalSize, actualRows, actualCols, actualSize;

            for(const auto matrix : {matrices...}) {
                
            }
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class uMemory, class uPadding,
                template<class, class> class rMemory, class rPadding>
        void _hostAddAvx(const Matrix<T, Memory, Padding> &a,
                         const Matrix<T, uMemory, uPadding> &b,
                         Matrix<T, rMemory, rPadding> &c,
                         uint64_t elemOffset, uint64_t maxElem) {
            T* _aPtr = a.getMatrix();
            T* _bPtr = b.getMatrix();
            T* _cPtr = c.getMatrix();

            uint64_t i = elemOffset;

            // implementation is defined for 32 bit and 64 bit AVX2
            if constexpr (std::is_floating_point_v<T>) {

                // self align logic
                bool isAligned = false;

                // double
                if constexpr (sizeof(T) == sizeof(double)) {
                    __m256d _a256, _b256, _c256;
                    __m128d _a128, _b128, _c128;

                    for(; i < (maxElem & ~0x3); i+=4) {
                        _a256 = _mm256_load_pd(&_aPtr[i]);
                        _b256 = _mm256_load_pd(&_bPtr[i]);

                        _c256 = _mm256_add_pd(_a256, _b256);
                        _mm256_stream_pd(&_cPtr[i], _c256);
                    }
                    for(; i < (maxElem & ~0x1); i+=2) {
                        _a128 = _mm_load_pd(&_aPtr[i]);
                        _b128 = _mm_load_pd(&_bPtr[i]);

                        _c128 = _mm_add_pd(_a128, _b128);
                        _mm_stream_pd(&_cPtr[i], _c128);
                    }
                } else if (sizeof(T) == sizeof(float)) /* float */ {
                    __m256 _a256, _b256, _c256;
                    __m128 _a128, _b128, _c128;

                    for(; i < (maxElem & ~0x5); i+=8) {
                        _a256 = _mm256_load_ps(&_aPtr[i]);
                        _b256 = _mm256_load_ps(&_bPtr[i]);

                        _c256 = _mm256_add_ps(_a256, _b256);
                        _mm256_stream_ps(&_cPtr[i], _c256);
                    }
                    for(; i < (maxElem & ~0x3); i+=4) {
                        _a128 = _mm_load_ps(&_aPtr[i]);
                        _b128 = _mm_load_ps(&_bPtr[i]);

                        _c128 = _mm_add_ps(_a128, _b128);
                        _mm_stream_ps(&_cPtr[i], _c128);
                    }
                }
            } else /* int */ {
                __m256i _a256, _b256, _c256;
                __m128i _a128, _b128, _c128;

                if constexpr (sizeof(T) == sizeof(int64_t)) {
                    for (; i < (maxElem & ~0x3); i += 4) {
                        _a256 = _mm256_load_si256((__m256i*)&_aPtr[i]);
                        _b256 = _mm256_load_si256((__m256i*)&_bPtr[i]);

                        _c256 = _mm256_add_epi64(_a256, _b256);
                        _mm256_stream_si256(&_cPtr[i], _c256);
                    }

                    for(; i < (maxElem & ~0x1); i+=2) {
                        _a128 = _mm_load_si128((__m128i*)&_aPtr[i]);
                        _b128 = _mm_load_si128((__m128i*)&_bPtr[i]);

                        _c128 = _mm_add_epi32(_a128, _b128);
                        _mm_stream_si128(&_cPtr[i], _c128);
                    }
                } else if constexpr (sizeof(T) == sizeof(int32_t)) {
//                    for (; i < (maxElem & ~0x5); i += 8) {
//                        _a256 = _mm256_load_si256((__m256i*)&_aPtr[i]);
//                        _b256 = _mm256_load_si256((__m256i*)&_bPtr[i]);
//
//                        _c256 = _mm256_add_epi32(_a256, _b256);
//                        _mm256_stream_si256((__m256i*)&_cPtr[i], _c256);
//                    }
//
//                    for(; i < (maxElem & ~0x3); i+=4) {
//                        _a128 = _mm_load_si128((__m128i*)_aPtr[i]);
//                        _b128 = _mm_load_si128((__m128i*)_bPtr[i]);
//
//                        _c128 = _mm_add_epi32(_a128, _b128);
//                        _mm_stream_si128((__m128i*)_cPtr[i], _c128);
//                    }
                    for (; i < (maxElem & ~0x7); i += 16) {
                        _a256 = _mm256_load_si256((__m256i*)&_aPtr[i]);
                        _b256 = _mm256_load_si256((__m256i*)&_bPtr[i]);

                        _c256 = _mm256_add_epi32(_a256, _b256);
                        _mm256_stream_si256((__m256i*)&_cPtr[i], _c256);

                        _a256 = _mm256_load_si256((__m256i*)&_aPtr[i+8]);
                        _b256 = _mm256_load_si256((__m256i*)&_bPtr[i+8]);

                        _c256 = _mm256_add_epi32(_a256, _b256);
                        _mm256_stream_si256((__m256i*)&_cPtr[i+8], _c256);
                    }

                    for(; i < (maxElem & ~0x5); i+=8) {
                        _a128 = _mm_load_si128((__m128i*)_aPtr[i+4]);
                        _b128 = _mm_load_si128((__m128i*)_bPtr[i+4]);

                        _c128 = _mm_add_epi32(_a128, _b128);
                        _mm_stream_si128((__m128i*)_cPtr[i], _c128);

                        _a128 = _mm_load_si128((__m128i*)_aPtr[i+4]);
                        _b128 = _mm_load_si128((__m128i*)_bPtr[i+4]);

                        _c128 = _mm_add_epi32(_a128, _b128);
                        _mm_stream_si128((__m128i*)_cPtr[i+4], _c128);
                    }
                } else if constexpr (sizeof(T) == sizeof(int16_t)) {
                    for (; i < (maxElem & ~0x7); i += 16) {
                        _a256 = _mm256_load_si256((__m256i*)_aPtr[i]);
                        _b256 = _mm256_load_si256((__m256i*)_bPtr[i]);

                        _c256 = _mm256_add_epi16(_a256, _b256);
                        _mm256_stream_si256((__m256i*)_cPtr[i], _c256);
                    }

                    for(; i < (maxElem & ~0x5); i+=8) {
                        _a128 = _mm_load_si128((__m128i*)_aPtr[i]);
                        _b128 = _mm_load_si128((__m128i*)_bPtr[i]);

                        _c128 = _mm_add_epi16(_a128, _b128);
                        _mm_stream_si128((__m128i*)_cPtr[i], _c128);
                    }
                } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                    for (; i < (maxElem & ~0x9); i += 32) {
                        _a256 = _mm256_load_si256((__m256i*)_aPtr[i]);
                        _b256 = _mm256_load_si256((__m256i*)_bPtr[i]);

                        _c256 = _mm256_add_epi8(_a256, _b256);
                        _mm256_stream_si256((__m256i*)_cPtr[i], _c256);
                    }

                    for(; i < (maxElem & ~0x7); i+=16) {
                        _a128 = _mm_load_si128((__m128i*)_aPtr[i]);
                        _b128 = _mm_load_si128((__m128i*)_bPtr[i]);

                        _c128 = _mm_add_epi8(_a128, _b128);
                        _mm_stream_si128((__m128i*)_cPtr[i], _c128);
                    }
                }
            }

            for(; i < maxElem; i++) {
                _cPtr[i] = _aPtr[i] +_bPtr[i];
            }
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class uMemory, class uPadding,
                template<class, class> class rMemory, class rPadding>
        void _addMatricesOnHost(const Matrix<T, Memory, Padding> &a,
                                const Matrix<T, uMemory, uPadding> &b,
                                Matrix<T, rMemory, rPadding> &c) requires
        IsDerivedFromHostMemoryBlock<T, Memory, Padding> &&
        IsDerivedFromHostMemoryBlock<T, uMemory, uPadding> &&
        IsDerivedFromHostMemoryBlock<T, rMemory, rPadding> {

            NaNL::ThreadPool *threadPool = NaNL::ThreadPool::getInstance();
            std::deque<std::future<void>> results;

            std::mutex print_mutex;

            /**
             *  Below is the constexpr branch for similar aligned matrices. This will use AVX/SSE intrinsics
             * assuming that all 3 matrices are aligned, and are the same padding.
             *
             * Else, launch
             */
            if constexpr (std::is_same_v<Padding, uPadding> && std::is_same_v<uPadding, rPadding>) {
                size_t cacheLineSize = std::hardware_constructive_interference_size;
                size_t typeSize = sizeof(T);
                size_t elemsPerCache = cacheLineSize / typeSize;

                // floor down, have a remainder
                uint64_t matrixSize = a.getActualTotalSize();
                uint64_t totalCacheLines = matrixSize / elemsPerCache;

                /**
                 * Below code is creating the total threads that will be used. Current logic assumes that
                 * there are more cache lines then CPUS.... else, only one thread will be used.
                 */
                uint64_t totalThreads = 4;//(threadPool->getAllocatedThreads() > totalCacheLines) ? 1 : threadPool->getAllocatedThreads();
                uint64_t threadCacheOffset = totalCacheLines / totalThreads;

                // initial start
                uint64_t currElemOffsetLow = 0;
                uint64_t elemOffsetJumpCount = threadCacheOffset * elemsPerCache;
                uint64_t currElemOffsetHigh = elemOffsetJumpCount;

                //std::cout << "Total AVX Threads: " << totalThreads << std::endl;

                for (uint64_t i = 0; i < totalThreads; i++) {

                    if (i != (totalThreads - 1)) {
                        std::future<void> result = threadPool->queue([&a, &b, &c, currElemOffsetLow, currElemOffsetHigh, &print_mutex] {
                                    _hostAddAvx(a, b, c, currElemOffsetLow, currElemOffsetHigh);
                                });
                        results.push_back(std::move(result));
                    } else {
                        std::future<void> result = threadPool->queue([&a, &b, &c, currElemOffsetLow, matrixSize] {
                            _hostAddAvx(a, b, c, currElemOffsetLow, matrixSize);
                        });
                        results.push_back(std::move(result));
                    }

                    currElemOffsetLow = currElemOffsetHigh;
                    currElemOffsetHigh += elemOffsetJumpCount;
                }
            } else {

                uint64_t desiredThreads = 4;//NaNL::ThreadPool::getInstance()->getAllocatedThreads();
                uint64_t totalThreads = a.getRows() >= desiredThreads ? desiredThreads : a.getRows();
                uint64_t rowsPerThread = a.getRows() / totalThreads;    // rounds down.
                uint64_t remainder = a.getRows() % totalThreads;
                uint64_t maxRowForThread = rowsPerThread + (remainder != 0 ? 1 : 0);

                uint64_t rowOffset = 0;
                for (uint64_t i = 0; i < totalThreads; i++) {
                    std::future<void> result = threadPool->queue([&a, &b, &c, rowOffset, maxRowForThread] {
                        for (uint64_t i = rowOffset; i < maxRowForThread; i++) {
                            for (uint64_t j = 0; j < a.getCols(); j++) {
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
            }

            /// wait for threads to finish.
            for (auto &result: results) {
                result.wait();
            }
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class uMemory, class uPadding,
                template<class, class> class rMemory, class rPadding>
        void _addMatricesOnCuda(const Matrix<T, Memory, Padding> &a,
                                const Matrix<T, uMemory, uPadding> &b,
                                Matrix<T, rMemory, rPadding> &c) requires
                IsDerivedFromDeviceMemoryBlock<T, Memory, Padding> &&
                IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding> &&
                IsDerivedFromDeviceMemoryBlock<T, rMemory, rPadding> {

            int currentDevice;
            cudaGetDevice(&currentDevice);

            const T *_a = a.getMatrix();
            const T *_b = b.getMatrix();
            T *_c = c.getMatrix();

            dim3 threadsPerBlock(512);

            if(a.getCudaDevice() != currentDevice && b.getCudaDevice() != currentDevice) {
                auto aa = a.template copyTo<Memory, Padding>();
                auto bb = b.template copyTo<Memory, Padding>();
                _a = aa.getMatrix();
                _b = bb.getMatrix();
                if(aa.getActualRows() == bb.getActualRows() &&
                        aa.getActualCols() == bb.getActualCols() &&
                        bb.getActualRows() == c.getActualRows() &&
                        bb.getActualCols() == c.getActualCols()) {
                    dim3 numBlocks(std::ceil((double)a.getActualTotalSize() / (double)threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                } else {
                    dim3 numBlocks(std::ceil((double)a.getTotalSize() / (double)threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatricesWithOffset<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, aa.getTotalSize(), aa.getRows(), aa.getCols(), aa.getActualCols(), bb.getActualCols(), c.getActualCols());
                }
            } else if(a.getCudaDevice() != currentDevice) {
                auto aa = a.template copyTo<Memory, Padding>();
                _a = aa.getMatrix();
                if(aa.getActualRows() == b.getActualRows() &&
                   aa.getActualCols() == b.getActualCols() &&
                   b.getActualRows() == c.getActualRows() &&
                   b.getActualCols() == c.getActualCols()) {
                    dim3 numBlocks(std::ceil((double) a.getActualTotalSize() / (double) threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                } else {
                    dim3 numBlocks(std::ceil((double)a.getTotalSize() / (double)threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatricesWithOffset<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getTotalSize(), a.getRows(), a.getCols(), aa.getActualCols(), b.getActualCols(), c.getActualCols());
                }

            } else if(b.getCudaDevice() != currentDevice) {
                auto bb = b.template copyTo<Memory, Padding>();
                _b = bb.getMatrix();
                if(a.getActualRows() == bb.getActualRows() &&
                   a.getActualCols() == bb.getActualCols() &&
                   bb.getActualRows() == c.getActualRows() &&
                   bb.getActualCols() == c.getActualCols()) {
                    dim3 numBlocks(std::ceil((double) a.getActualTotalSize() / (double) threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                } else {
                    dim3 numBlocks(std::ceil((double)a.getTotalSize() / (double)threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatricesWithOffset<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getTotalSize(), a.getRows(), a.getCols(), a.getActualCols(), bb.getActualCols(), c.getActualCols());
                }
            } else {
                if(a.getActualRows() == b.getActualRows() &&
                   a.getActualCols() == b.getActualCols() &&
                   b.getActualRows() == c.getActualRows() &&
                   b.getActualCols() == c.getActualCols()) {
                    dim3 numBlocks(std::ceil((double) a.getActualTotalSize() / (double) threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatrices<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getActualTotalSize());
                } else {
                    dim3 numBlocks(std::ceil((double)a.getTotalSize() / (double)threadsPerBlock.x));
                    Internal::Kernels::deviceAddMatricesWithOffset<<<numBlocks, threadsPerBlock>>>(_a, _b, _c, a.getTotalSize(), a.getRows(), a.getCols(), a.getActualCols(), b.getActualCols(), c.getActualCols());

                }
            }

            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        template<class T, template<class, class> class Memory, class Padding,
                class R>
        static Matrix<R, Memory, Padding> _castMatricesOnCuda(const Matrix<T, Memory, Padding> &a) //requires
            //IsDerivedFromDeviceMemoryBlock<T, Memory, Padding> &&
            //IsDerivedFromDeviceMemoryBlock<R, Memory, Padding>
        {
            // fetch current cuda device number.
            int deviceNum;
            gpuErrchk(cudaGetDevice(&deviceNum));
            const int currentCudaDevice = deviceNum;

            // set cuda to 'a matrix' device.
            cudaSetDevice(a.getCudaDevice());
            auto castMatrix = Matrix<R, Memory, Padding>(a.getRows(), a.getCols());


            dim3 threadsPerBlock(1024);
            dim3 numBlocks(std::ceil((double)a.getActualTotalSize() / (double)threadsPerBlock.x) );
            NaNL::Internal::Kernels::deviceMatrixCast<<<numBlocks, threadsPerBlock>>>(a.getMatrix(), castMatrix.getMatrix(), a.getActualTotalSize());

            gpuErrchk(cudaDeviceSynchronize());

            // set cuda device back to current.
            gpuErrchk(cudaSetDevice(currentCudaDevice));
            return castMatrix;
        }


        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding, class R>
        static Matrix<R, rMemory, rPadding> _copyHostToDevice(Matrix<T, Memory, Padding> &src) {
            Matrix<R, rMemory, rPadding> copyMatrix(src.getRows(), src.getCols());

            if (src.getActualRows() == copyMatrix.getActualRows()
                && src.getActualCols() == copyMatrix.getActualCols()
                && src.getActualTotalSize() == copyMatrix.getActualTotalSize()) {
                if constexpr (std::is_same_v<T,R>) {
                    cudaMemcpy(copyMatrix.getMatrix(), src.getMatrix(), src.getActualTotalSize() * sizeof(T),
                               cudaMemcpyHostToDevice);
                } else {
                    auto castMatrix = src.template copyTo<Memory, Padding, R>();
                    cudaMemcpy(copyMatrix.getMatrix(), castMatrix.getMatrix(), castMatrix.getActualTotalSize() * sizeof(R),
                               cudaMemcpyHostToDevice);
                }
            } else {
                if constexpr(std::is_same_v<T,R>) {
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             src.getMatrix() + src.getActualCols() * i,
                                             src.getCols() * sizeof(T),
                                             cudaMemcpyHostToDevice));
                    }
                } else {
                    auto castMatrix = src.template copyTo<Memory, Padding, R>();
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             castMatrix.getMatrix() + castMatrix.getActualCols() * i,
                                             castMatrix.getCols() * sizeof(R),
                                             cudaMemcpyHostToDevice));
                    }
                }
            }

            return copyMatrix;
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding, class R>
        static Matrix<R, rMemory, rPadding> _copyDeviceToHost(Matrix<T, Memory, Padding> &src) {
            Matrix<R, rMemory, rPadding> copyMatrix(src.getRows(), src.getCols());

            if (src.getActualRows() == copyMatrix.getActualRows()
                && src.getActualCols() == copyMatrix.getActualCols()
                && src.getActualTotalSize() == copyMatrix.getActualTotalSize()) {
                if constexpr (std::is_same_v<T,R>) {
                    cudaMemcpy(copyMatrix.getMatrix(), src.getMatrix(), src.getActualTotalSize() * sizeof(T),
                               cudaMemcpyDeviceToHost);
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Padding, R>(src);
                    cudaMemcpy(copyMatrix.getMatrix(), castMatrix.getMatrix(), castMatrix.getActualTotalSize() * sizeof(R),
                               cudaMemcpyDeviceToHost);
                }
            } else {
                if constexpr (std::is_same_v<T,R>) {
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             src.getMatrix() + src.getActualCols() * i,
                                             src.getCols() * sizeof(T),
                                             cudaMemcpyDeviceToHost));
                    }
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Padding, R>(src);
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             castMatrix.getMatrix() + castMatrix.getActualCols() * i,
                                             castMatrix.getCols() * sizeof(R),
                                             cudaMemcpyDeviceToHost));
                    }
                }
            }

            return copyMatrix;
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding>
        static void _copyDeviceToDeviceStride(Matrix<T, Memory, Padding> &src, Matrix<T, rMemory, rPadding> &dst) {
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

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding>
        static void _copyPeerToPeerStride(Matrix<T, Memory, Padding> &src, Matrix<T, rMemory, rPadding> &dst) {
            std::vector<cudaStream_t> streams;
            streams.reserve(src.getRows());
            int srcDevice = src.getCudaDevice();
            int dstDevice = dst.getCudaDevice();

            // async device to device memcpy.
            for (uint64_t i = 0; i < dst.getRows(); i++) {
                streams.emplace_back();
                gpuErrchk(cudaMemcpyPeerAsync(dst.getMatrix() + dst.getActualCols() * i,
                                          dstDevice,
                                          src.getMatrix() + src.getActualCols() * i,
                                          srcDevice,
                                          src.getCols() * sizeof(T),
                                          cudaMemcpyDeviceToDevice, streams[i]));
            }

            // sync and cleanup
            for(cudaStream_t stream: streams) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding>
        static void _copyDeviceToDeviceContinous(Matrix<T, Memory, Padding> &src, Matrix<T, rMemory, rPadding> &dst) {
            gpuErrchk(cudaMemcpy(dst.getMatrix(), src.getMatrix(), src.getActualTotalSize() * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding>
        static void _copyPeerToPeerContinous(Matrix<T, Memory, Padding> &src, Matrix<T, rMemory, rPadding> &dst) {
            int srcDevice = src.getCudaDevice();
            int dstDevice = dst.getCudaDevice();
            gpuErrchk(cudaMemcpyPeer(dst.getMatrix(), dstDevice, src.getMatrix(), srcDevice, src.getActualTotalSize() * sizeof(T)));
        }

        template<class T, template<class, class> class Memory, class Padding,
                template<class, class> class rMemory, class rPadding, class R>
        static Matrix<R, rMemory, rPadding> _copyDeviceToDevice(Matrix<T, Memory, Padding> &src) {
            Matrix<R, rMemory, rPadding> copyMatrix(src.getRows(), src.getCols());
            bool isSameDevice = copyMatrix.getCudaDevice() == src.getCudaDevice() ? true : false;

            // if actual size of both matrices are the same. 1:1 copy.
            // else copy each row in respect of their Padding.
            if (src.getActualRows() == copyMatrix.getActualRows()
                && src.getActualCols() == copyMatrix.getActualCols()) {
                if constexpr (std::is_same_v<T, R>) {
                    if(isSameDevice) {
                        Internal::_copyDeviceToDeviceContinous(src, copyMatrix);
                    } else {
                        _copyPeerToPeerContinous(src, copyMatrix);
                    }
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Padding, R>(src);
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
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Padding, R>(src);
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