//
// Created by steve on 11/27/2022.
//
#include "Matrix.cuh"


namespace NaNL {
//    template<class T, template<class, class> class Memory,
//            class Alignment>
//    Matrix<T, Memory, Alignment>::Matrix()
//            : BaseMatrix<T, Memory, Alignment>(0, 0) {
//
//    }

    template<class T, template<class, class> class Memory,
            class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(uint64_t rows, uint64_t cols)
            : BaseMatrix<T, Memory, Alignment>(rows, cols) {
    }

    template<class T, template<class, class> class Memory, class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(const Matrix<T, Memory, Alignment> &copyMatrix) noexcept
            : BaseMatrix<T, Memory, Alignment>(copyMatrix.getRows(), copyMatrix.getCols()) {
        /*
         * TODO: Fix this method. Doesn't copy properly.
         */
        if constexpr (Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>) {
            for (uint64_t i = 0; i < this->rows; i++) {
                for (uint64_t j = 0; j < this->cols; j++) {
                    this->operator[](i)[j] = copyMatrix.get(i, j);
                }
            }
        } else {
            cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    template<class T, template<class, class> class Memory, class Alignment>
    Matrix<T, Memory, Alignment>::Matrix(Matrix<T, Memory, Alignment> &&copyMatrix) noexcept
            : BaseMatrix<T, Memory, Alignment>(copyMatrix.rows, copyMatrix.cols) {
        this->_matrix = std::move(copyMatrix._matrix);
    }

    /*
     * TODO: provide implementation.
     */
    template<class T, template<class, class> class Memory, class Alignment>
    NaNL::Matrix<T, Memory, Alignment> &
    NaNL::Matrix<T, Memory, Alignment>::Matrix::operator=(const Matrix<T, Memory, Alignment> &rhs) {
        std::unique_lock<std::shared_mutex> lock(this->_shared_mutex);
        return *this;
    }

//    template<class T, template<class, class> class Memory, class Alignment>
//    template<template<class, class> class rMemory, class rAlignment>
//    Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::copyTo() const {
//        Matrix<T, rMemory, rAlignment> copyMatrix(this->getRows(), this->getCols());
//        cudaMemcpyKind memcpyKind = cudaMemcpyHostToHost;
//
//        // host to host
//        if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
//                        && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
//            for (uint64_t i = 0; i < this->getRows(); i++) {
//                for (uint64_t j = 0; j < this->getCols(); j++) {
//                    copyMatrix[i][j] = this->get(i, j);
//                }
//            }
//            return copyMatrix;
//        } // host to device
//        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
//                        && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>) {
//            memcpyKind = cudaMemcpyHostToDevice;
//        } // device to host
//        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Alignment>
//                            && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
//            memcpyKind = cudaMemcpyDeviceToHost;
//        } // device to device
//        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Alignment>
//                            && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>) {
//            memcpyKind = cudaMemcpyDeviceToDevice;
//        }
//
//        // Perform copy based on kind of copy decided from above.
//        // If the actual total size(for different alignments) is
//        // the same, we can perform a 1:1 copy rather a loop-based
//        // copy over.
//        if (this->getActualRows() == copyMatrix.getActualRows()
//            && this->getActualCols() == copyMatrix.getActualCols()
//            && this->getActualTotalSize() == copyMatrix.getActualTotalSize()) {
//            cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T),
//                       memcpyKind);
//        } else {
//            for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
//                gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
//                                     this->getMatrix() + this->getActualCols() * i,
//                                     this->getActualCols() * sizeof(T),
//                                     memcpyKind));
//            }
//        }
//
//        return copyMatrix;
//    }


    template<class T, template<class, class> class Memory, class Alignment>
    std::shared_mutex &Matrix<T, Memory, Alignment>::getMutex() const {
        return this->_shared_mutex;
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment, class R>
    Matrix<R, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::copyTo() const {
        std::shared_lock<std::shared_mutex> lock(this->_shared_mutex);
        Matrix<R, rMemory, rAlignment> copyMatrix(this->getRows(), this->getCols());

        // host to host
        if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
                     && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
            for (uint64_t i = 0; i < this->getRows(); i++) {
                for (uint64_t j = 0; j < this->getCols(); j++) {
                    // if R and T are same, no need to cast, pretty sure the compiler would recognize
                    // that T and R are the same, or once deduced to int -> int, no need for cast.
                    // But just in case :)
                    if constexpr (std::is_same_v<T,R>) {
                        copyMatrix[i][j] = this->get(i, j);
                    } else {
                        copyMatrix[i][j] = (R)this->get(i, j);
                    }
                }
            }
            return copyMatrix;
        } // host to device
        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
                          && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>) {
            if (this->getActualRows() == copyMatrix.getActualRows()
                && this->getActualCols() == copyMatrix.getActualCols()
                && this->getActualTotalSize() == copyMatrix.getActualTotalSize()) {
                if constexpr (std::is_same_v<T,R>) {
                    cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T),
                               cudaMemcpyHostToDevice);
                } else {
                    auto castMatrix = this->copyTo<Memory, Alignment, R>();
                    cudaMemcpy(copyMatrix.getMatrix(), castMatrix.getMatrix(), castMatrix.getActualTotalSize() * sizeof(R),
                               cudaMemcpyHostToDevice);
                }
            } else {
                if constexpr(std::is_same_v<T,R>) {
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             this->getMatrix() + this->getActualCols() * i,
                                             this->getCols() * sizeof(T),
                                             cudaMemcpyHostToDevice));
                    }
                } else {
                    auto castMatrix = this->copyTo<Memory, Alignment, R>();
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             castMatrix.getMatrix() + castMatrix.getActualCols() * i,
                                             castMatrix.getCols() * sizeof(R),
                                             cudaMemcpyHostToDevice));
                    }
                }
            }
        } // device to host
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Alignment>
                          && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
            if (this->getActualRows() == copyMatrix.getActualRows()
                && this->getActualCols() == copyMatrix.getActualCols()
                && this->getActualTotalSize() == copyMatrix.getActualTotalSize()) {
                if constexpr (std::is_same_v<T,R>) {
                    cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T),
                               cudaMemcpyDeviceToHost);
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Alignment, R>(*this);
                    cudaMemcpy(copyMatrix.getMatrix(), castMatrix.getMatrix(), castMatrix.getActualTotalSize() * sizeof(R),
                               cudaMemcpyDeviceToHost);
                }
            } else {
                if constexpr (std::is_same_v<T,R>) {
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             this->getMatrix() + this->getActualCols() * i,
                                             this->getCols() * sizeof(T),
                                             cudaMemcpyDeviceToHost));
                    }
                } else {
                    auto castMatrix = Internal::_castMatricesOnCuda<T, Memory, Alignment, R>(*this);
                    for (uint64_t i = 0; i < copyMatrix.getRows(); i++) {
                        gpuErrchk(cudaMemcpy(copyMatrix.getMatrix() + copyMatrix.getActualCols() * i,
                                             castMatrix.getMatrix() + castMatrix.getActualCols() * i,
                                             castMatrix.getCols() * sizeof(R),
                                             cudaMemcpyDeviceToHost));
                    }
                }
            }
        } // device to device
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Alignment>
                          && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>) {
            //Internal::testAgain<T>(100);
            return Internal::_copyDeviceToDevice<T, Memory, Alignment, rMemory, rAlignment, R>((Matrix<T, Memory, Alignment> &)(*this));
        }

        return copyMatrix;
    }




    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment>
    Matrix<T, rMemory, rAlignment> NaNL::Matrix<T, Memory, Alignment>::moveTo() const && {
        std::shared_lock<std::shared_mutex> lock(this->_shared_mutex);

        // if moving to the same thing, just return.
        if constexpr(std::is_base_of_v<Matrix<T, Memory, Alignment>, Matrix<T, rMemory, rAlignment>>) {
            if constexpr (Internal::is_matrix_derived_from_device<T, Memory, Alignment>
                    && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>) {
                // check if they exist on same device.
                int currentDevice;
                gpuErrchk(cudaGetDevice(&currentDevice));
                if(this->getCudaDevice() != currentDevice) {
                    Matrix<T, rMemory, rAlignment> movedMatrix(this->getRows(), this->getCols());
                    if(this->getActualRows() == movedMatrix.getActualCols()
                       && this->getActualCols() == movedMatrix.getActualCols()) {
                        cudaMemcpyPeer(movedMatrix.getMatrix(), movedMatrix.getCudaDevice(), this->getMatrix(), this->getCudaDevice(), sizeof(T) * this->getActualTotalSize());
                    } else {
                        for(uint64_t i = 0; i < this->getRows(); i++) {
                            gpuErrchk(cudaMemcpyPeer(movedMatrix.getMatrix(), movedMatrix.getCudaDevice(), this->getMatrix(), this->getCudaDevice(), sizeof(T) * this->getActualTotalSize()));
                        }
                    }
                    return movedMatrix;
                } else {
                    return *this;
                }
            } else {
                return *this;
            }
        }

        // If this is Matrix is stored on host memory, and if the desired Matrix
        // is also stored on host memory.
        else if constexpr(NaNL::Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
                && NaNL::Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
            Matrix<T, rMemory, rAlignment> movedMatrix(this->getRows(), this->getCols());
            for(uint64_t i = 0; i < this->getRows(); i++) {
                for(uint64_t j = 0; j < this->getCols(); j++) {
                    movedMatrix[i][j] = this->get(i,j);
                }
            }

            return movedMatrix;
        }
        // Matrix is from Host to Device
        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Alignment>
                && Internal::is_matrix_derived_from_device<T, rMemory, rAlignment>){
            Matrix<T, rMemory, rAlignment> movedMatrix(this->getRows(), this->getCols());
            if(this->getActualRows() == movedMatrix.getActualRows()
                && this->getActualCols() == movedMatrix.getActualCols()) {
                cudaMemcpy(movedMatrix.getMatrix(), this->getMatrix(), sizeof(T) * this->getActualTotalSize(), cudaMemcpyHostToDevice);
            } else {
                for(uint64_t i = 0; i < this->getRows(); i++) {
                    gpuErrchk(cudaMemcpy(movedMatrix.getMatrix() + movedMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T), cudaMemcpyHostToDevice));
                }
            }

            return movedMatrix;
        }
        // Matrix is from Device to Host
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Alignment>
                && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rAlignment>) {
            Matrix<T, rMemory, rAlignment> movedMatrix(this->getRows(), this->getCols());
            if(this->getActualRows() == movedMatrix.getActualRows()
               && this->getActualCols() == movedMatrix.getActualCols()) {
                cudaMemcpy(movedMatrix.getMatrix(), this->getMatrix(), sizeof(T) * this->getActualTotalSize(), cudaMemcpyDeviceToHost);
            } else {
                for(uint64_t i = 0; i < movedMatrix.getRows(); i++) {
                    gpuErrchk(cudaMemcpy(movedMatrix.getMatrix() + movedMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getActualCols() * sizeof(T), cudaMemcpyDeviceToHost));
                }
            }

            return movedMatrix;
        } else {
            []<bool flag = false>() {
                static_assert(flag, "There is no implementation for the Matrix::moveTo for this combination of types.");
            };
        }
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment,
            template<class, class> class uMemory, class uAlignment>
    Matrix<T, rMemory, rAlignment>
    Matrix<T, Memory, Alignment>::add(const Matrix<T, uMemory, uAlignment> &b, MatrixAddOperation device) {
        std::shared_lock<std::shared_mutex> aLock(this->getMutex());
        std::shared_lock<std::shared_mutex> bLock(b.getMutex());

        if (MatrixAddOperation::TensorCores == device) {

        } else if (MatrixAddOperation::Cuda == device) {
            return addCuda<rMemory, rAlignment, uMemory, uAlignment>((const Matrix<T, Memory, Alignment> &) (*this), b);
        } else /*Host*/ {
            //return addHost<T, rMemory, rAlignment>((const Matrix<T, Memory, Alignment> &) (*this), b);
            return addHost<rMemory, rAlignment, uMemory, uAlignment>((const Matrix<T, Memory, Alignment> &) (*this), b);
        }
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment,
            template<class, class> class uMemory, class uAlignment>
    Matrix<T, rMemory, rAlignment> Matrix<T, Memory, Alignment>::addHost(const Matrix<T, Memory, Alignment> &a,
                                                          const Matrix<T, uMemory, uAlignment> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }

        // Since C is to be calculated on host, define as paged memory with return alignment.
        Matrix<T, PagedMemoryBlock, rAlignment> c(a.getRows(), a.getCols());

        // a & b are both derived from HostMemoryBlock
        if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                && IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnHost(a, b, c);
        }

        // Only a is derived from HostMemoryBlock
        else if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                && !IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnHost(a, b.template copyTo<PagedMemoryBlock, uAlignment>(), c);
        }

        // Only b is derived from HostMemoryBlock
        else if constexpr(!IsDerivedFromHostMemoryBlock<T, Memory, Alignment>
                && IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnHost(a.template copyTo<PagedMemoryBlock, Alignment>(), b, c);
        }

        // Neither a or b derived from HostMemoryBlock
        else {
            Internal::_addMatricesOnHost(a.template copyTo<PagedMemoryBlock, Alignment>(),
                                         b.template copyTo<PagedMemoryBlock, uAlignment>(), c);
        }

        // TODO: move c to requested
        return std::move(c).template moveTo<rMemory, rAlignment>();
    }

    template<class T, template<class, class> class Memory, class Alignment>
    template<template<class, class> class rMemory, class rAlignment,
            template<class, class> class uMemory, class uAlignment>
    Matrix<T, rMemory, rAlignment> Matrix<T, Memory, Alignment>::addCuda(const Matrix<T, Memory, Alignment> &a,
                                                                         const Matrix<T, uMemory, uAlignment> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }

        // Since C is to be calculated on host, define as paged memory with return alignment.
        std::unique_ptr<Matrix<T, DeviceMemoryBlock, rAlignment>> aDevicePtr;
        std::unique_ptr<Matrix<T, DeviceMemoryBlock, rAlignment>> bDevicePtr;


        Matrix<T, DeviceMemoryBlock, rAlignment> c(a.getRows(), a.getCols());

        // a & b are both derived from HostMemoryBlock
        if constexpr(IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment>
                     && IsDerivedFromDeviceMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnCuda(a, b, c);
        }

        // Only a is derived from HostMemoryBlock
        else if constexpr(IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment>
                          && !IsDerivedFromDeviceMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnCuda(a, b.template copyTo<DeviceMemoryBlock, uAlignment>(), c);
        }

        // Only b is derived from HostMemoryBlock
        else if constexpr(!IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment>
                          && IsDerivedFromDeviceMemoryBlock<T, uMemory, uAlignment>) {
            Internal::_addMatricesOnCuda(a.template copyTo<DeviceMemoryBlock, Alignment>(), b, c);
        }

        // Neither a or b derived from HostMemoryBlock
        else {
            Internal::_addMatricesOnCuda(a.template copyTo<DeviceMemoryBlock, Alignment>(),
                                         b.template copyTo<DeviceMemoryBlock, uAlignment>(), c);
        }

        // TODO: move c to requested
        return std::move(c).template moveTo<rMemory, rAlignment>();
    }
//    template<class T, template<class, class> class Memory, class Alignment>
//    template<template<class, class> class uMemory, class uAlignment,
//        template<class, class> class rMemory, class rAlignment>
//    Matrix<T, rMemory, rAlignment> Matrix<T, Memory, Alignment>::addCuda(const Matrix<T, Memory, Alignment> &a, const Matrix<T, uMemory, uAlignment> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }
//
//        return Matrix<T, rMemory, rAlignment>();
//    }
}


//template<class T>
//NaNL::Matrix<T,NaNL::PagedMemoryBlock, NaNL::Unaligned>::Matrix(uint64_t rows, uint64_t cols) : BaseMatrix<T, NaNL::PagedMemoryBlock, NaNL::Unaligned>(rows, cols)
//{
//
//}
//
//
//template<class T>
//void NaNL::Matrix<T,NaNL::PagedMemoryBlock,NaNL::Unaligned>::add(const BaseMatrix <T, NaNL::PagedMemoryBlock, NaNL::Unaligned> &b, NaNL::MatrixDeviceOperation device) {
//    std::cout << "The one we want." << std::endl;
//}
//
//template<class T>
//void NaNL::Matrix<T,NaNL::PinnedMemoryBlock,NaNL::Unaligned>::add(const BaseMatrix <T, NaNL::PinnedMemoryBlock, NaNL::Unaligned> &b, NaNL::MatrixDeviceOperation device) {
//    std::cout << "The one we don't want." << std::endl;
//}

//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(unsigned long numberOfRows, unsigned long numberOfCols) :
//        BaseMatrix<T>(numberOfRows, numberOfCols, _freePagedMemory) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    _allocateMemory(numberOfRows, numberOfCols);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(const NaNL::Matrix<T, NaNL::Device::Host> &copyMatrix) noexcept :
//BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePagedMemory) { ; }
//
//
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host>::Matrix(NaNL::Matrix<T, NaNL::Device::Host> &&copyMatrix)  noexcept :
//BaseMatrix<T>(copyMatrix.rows, copyMatrix.cols, _freePagedMemory) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    this->matrix = std::move(copyMatrix.matrix);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::_allocateMemory(unsigned long rows, unsigned long cols)
//{
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    T* _pagedArr = new T[this->totalSize];
//    this->matrix = std::unique_ptr<T[], void(*)(T*)>(_pagedArr, _freePagedMemory);
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::_hostAddMatrices(T* _a, T* _b, T* _c, unsigned long _blockSize, unsigned long _offset) {
//    for(unsigned long i = 0; i < _blockSize; i++) {
//        _c[_offset + i] = _a[_offset + i] + _b[_offset + i];
//    }
//}
//
//template<typename T>
//void NaNL::Matrix<T, NaNL::Device::Host>::add(const NaNL::BaseMatrix<T> &bMatrix) {
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_START;
//#endif
//
//    if(this->validateMatricesAreSameShape(bMatrix)) {
//        //throw new MatrixIsInvalidShape();
//    }
//
//    T* a = this->matrix.get();
//    T* b = bMatrix.matrix.get();
//    T* c = this->matrix.get();
//
//    // get total totalThreads.
//    unsigned long totalThreads = NaNL::ThreadPool::getInstance()->getAllocatedThreads();
//
//    // calculate block size per thread.
//    unsigned long blockSize = this->totalSize / totalThreads;
//    unsigned long remainder = this->totalSize - blockSize * totalThreads;
//    unsigned long threadOffset = 0;
//
//    NaNL::ThreadPool* threadPool = NaNL::ThreadPool::getInstance();
//    std::deque<std::future<void>> results;
//
//    for(unsigned long i = 0; i < totalThreads; i++) {
//        unsigned long modifiedBlockSize = (remainder == 0) ? blockSize : blockSize + 1;
//        std::future<void> result = threadPool->queue([this, a, b, c, modifiedBlockSize, threadOffset] { _hostAddMatrices(a,b,c, modifiedBlockSize, threadOffset); });
//        results.push_back(std::move(result));
//        threadOffset += modifiedBlockSize;
//
//        if(remainder != 0) {
//            remainder--;
//        }
//    }
//
//    // wait on all futures to be populated/deferred.
//    for(auto & result : results) {
//        result.wait();
//    }
//
//#ifdef PERFORMANCE_LOGGING
//    PERFORMANCE_LOGGING_END;
//#endif
//}
//
//template<typename T>
//NaNL::Matrix<T, NaNL::Device::Host> &
//NaNL::Matrix<T, NaNL::Device::Host>::operator=(const NaNL::Matrix<T, NaNL::Device::Host> &rhs) {
//    return *this;
//}





