//
// Created by steve on 11/27/2022.
//
#include "Matrix.cuh"


namespace NaNL {
//    template<class T, template<class, class> class Memory,
//            class Padding>
//    Matrix<T, Memory, Padding>::Matrix()
//            : BaseMatrix<T, Memory, Padding>(0, 0) {
//
//    }



    template<class T, template<class, class> class Memory,
            class Padding>
    Matrix<T, Memory, Padding>::Matrix(uint64_t rows, uint64_t cols)
            : BaseMatrix<T, Memory, Padding>(rows, cols) {
    }

    template<class T, template<class, class> class Memory, class Padding>
    Matrix<T, Memory, Padding>::Matrix(const Matrix<T, Memory, Padding> &copyMatrix) noexcept
            : BaseMatrix<T, Memory, Padding>(copyMatrix.getRows(), copyMatrix.getCols()) {
        /*
         * TODO: Fix this method. Doesn't copy properly.
         */
        if constexpr (Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>) {
            for (uint64_t i = 0; i < this->rows; i++) {
                for (uint64_t j = 0; j < this->cols; j++) {
                    this->operator[](i)[j] = copyMatrix.get(i, j);
                }
            }
        } else {
            cudaMemcpy(copyMatrix.getMatrix(), this->getMatrix(), this->getActualTotalSize() * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    template<class T, template<class, class> class Memory, class Padding>
    Matrix<T, Memory, Padding>::Matrix(Matrix<T, Memory, Padding> &&copyMatrix) noexcept
            : BaseMatrix<T, Memory, Padding>(copyMatrix.rows, copyMatrix.cols) {
        this->_matrix = std::move(copyMatrix._matrix);
    }

    /*
     * TODO: provide implementation.
     */
    template<class T, template<class, class> class Memory, class Padding>
    NaNL::Matrix<T, Memory, Padding> &
    NaNL::Matrix<T, Memory, Padding>::Matrix::operator=(const Matrix<T, Memory, Padding> &rhs) {
        std::unique_lock<std::shared_mutex> lock(this->_shared_mutex);
        return *this;
    }

//    template<class T, template<class, class> class Memory, class Padding>
//    template<template<class, class> class rMemory, class rPadding>
//    Matrix<T, rMemory, rPadding> NaNL::Matrix<T, Memory, Padding>::copyTo() const {
//        Matrix<T, rMemory, rPadding> copyMatrix(this->getRows(), this->getCols());
//        cudaMemcpyKind memcpyKind = cudaMemcpyHostToHost;
//
//        // host to host
//        if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
//                        && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
//            for (uint64_t i = 0; i < this->getRows(); i++) {
//                for (uint64_t j = 0; j < this->getCols(); j++) {
//                    copyMatrix[i][j] = this->get(i, j);
//                }
//            }
//            return copyMatrix;
//        } // host to device
//        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
//                        && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>) {
//            memcpyKind = cudaMemcpyHostToDevice;
//        } // device to host
//        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Padding>
//                            && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
//            memcpyKind = cudaMemcpyDeviceToHost;
//        } // device to device
//        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Padding>
//                            && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>) {
//            memcpyKind = cudaMemcpyDeviceToDevice;
//        }
//
//        // Perform copy based on kind of copy decided from above.
//        // If the actual total size(for different Paddings) is
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


    template<class T, template<class, class> class Memory, class Padding>
    std::shared_mutex &Matrix<T, Memory, Padding>::getMutex() const {
        return this->_shared_mutex;
    }

    template<class T, template<class, class> class Memory, class Padding>
    template<template<class, class> class rMemory, class rPadding, class R>
    Matrix<R, rMemory, rPadding> NaNL::Matrix<T, Memory, Padding>::copyTo() const {
        std::shared_lock<std::shared_mutex> lock(this->_shared_mutex);
        Matrix<R, rMemory, rPadding> copyMatrix(this->getRows(), this->getCols());

        // host to host
        if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
                     && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
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
        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
                          && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>) {
            return Internal::_copyHostToDevice<T, Memory, Padding, rMemory, rPadding, R>((Matrix<T, Memory, Padding> &)(*this));
        } // device to host
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Padding>
                          && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
            return Internal::_copyDeviceToHost<T, Memory, Padding, rMemory, rPadding, R>((Matrix<T, Memory, Padding> &)(*this));
        } // device to device
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Padding>
                          && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>) {
            //Internal::testAgain<T>(100);
            return Internal::_copyDeviceToDevice<T, Memory, Padding, rMemory, rPadding, R>((Matrix<T, Memory, Padding> &)(*this));
        }

        return copyMatrix;
    }




    template<class T, template<class, class> class Memory, class Padding>
    template<template<class, class> class rMemory, class rPadding>
    Matrix<T, rMemory, rPadding> NaNL::Matrix<T, Memory, Padding>::moveTo() const && {
        std::shared_lock<std::shared_mutex> lock(this->_shared_mutex);

        // if moving to the same thing, just return.
        if constexpr(std::is_base_of_v<Matrix<T, Memory, Padding>, Matrix<T, rMemory, rPadding>>) {
            if constexpr (Internal::is_matrix_derived_from_device<T, Memory, Padding>
                    && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>) {
                // check if they exist on same device.
                int currentDevice;
                gpuErrchk(cudaGetDevice(&currentDevice));
                if(this->getCudaDevice() != currentDevice) {
                    Matrix<T, rMemory, rPadding> movedMatrix(this->getRows(), this->getCols());
                    if(this->getActualRows() == movedMatrix.getActualRows()
                       && this->getActualCols() == movedMatrix.getActualCols()) {
                        cudaMemcpyPeer(movedMatrix.getMatrix(), movedMatrix.getCudaDevice(), this->getMatrix(), this->getCudaDevice(), sizeof(T) * this->getActualTotalSize());
                    } else {
                        for(uint64_t i = 0; i < this->getRows(); i++) {
                            gpuErrchk(cudaMemcpyPeer(movedMatrix.getMatrix() + movedMatrix.getActualCols() * i, movedMatrix.getCudaDevice(),
                                                     this->getMatrix() + this->getActualCols() * i, this->getCudaDevice(),
                                                     sizeof(T) * this->getCols()));
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
        else if constexpr(NaNL::Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
                && NaNL::Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
            Matrix<T, rMemory, rPadding> movedMatrix(this->getRows(), this->getCols());
            for(uint64_t i = 0; i < this->getRows(); i++) {
                for(uint64_t j = 0; j < this->getCols(); j++) {
                    movedMatrix[i][j] = this->get(i,j);
                }
            }

            return movedMatrix;
        }
        // Matrix is from Host to Device
        else if constexpr(Internal::is_matrix_derived_from_paged_or_pinned<T, Memory, Padding>
                && Internal::is_matrix_derived_from_device<T, rMemory, rPadding>){
            Matrix<T, rMemory, rPadding> movedMatrix(this->getRows(), this->getCols());
            if(this->getActualRows() == movedMatrix.getActualRows()
                && this->getActualCols() == movedMatrix.getActualCols()) {
                cudaMemcpy(movedMatrix.getMatrix(), this->getMatrix(), sizeof(T) * this->getActualTotalSize(), cudaMemcpyHostToDevice);
            } else {
                for(uint64_t i = 0; i < this->getRows(); i++) {
                    gpuErrchk(cudaMemcpy(movedMatrix.getMatrix() + movedMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getCols() * sizeof(T), cudaMemcpyHostToDevice));
                }
            }

            return movedMatrix;
        }
        // Matrix is from Device to Host
        else if constexpr(Internal::is_matrix_derived_from_device<T, Memory, Padding>
                && Internal::is_matrix_derived_from_paged_or_pinned<T, rMemory, rPadding>) {
            Matrix<T, rMemory, rPadding> movedMatrix(this->getRows(), this->getCols());
            if(this->getActualRows() == movedMatrix.getActualRows()
               && this->getActualCols() == movedMatrix.getActualCols()) {
                cudaMemcpy(movedMatrix.getMatrix(), this->getMatrix(), sizeof(T) * this->getActualTotalSize(), cudaMemcpyDeviceToHost);
            } else {
                for(uint64_t i = 0; i < movedMatrix.getRows(); i++) {
                    gpuErrchk(cudaMemcpy(movedMatrix.getMatrix() + movedMatrix.getActualCols() * i, this->getMatrix() + this->getActualCols() * i, this->getCols() * sizeof(T), cudaMemcpyDeviceToHost));
                }
            }

            return movedMatrix;
        } else {
            []<bool flag = false>() {
                static_assert(flag, "There is no implementation for the Matrix::moveTo for this combination of types.");
            };
        }
    }

    template<class T, template<class, class> class Memory, class Padding>
    template<template<class, class> class rMemory, class rPadding,
            template<class, class> class uMemory, class uPadding>
    Matrix<T, rMemory, rPadding>
    Matrix<T, Memory, Padding>::add(const Matrix<T, uMemory, uPadding> &b, MatrixAddOperation device) {
        std::shared_lock<std::shared_mutex> aLock(this->getMutex());
        std::shared_lock<std::shared_mutex> bLock(b.getMutex());

        if (MatrixAddOperation::TensorCores == device) {

        } else if (MatrixAddOperation::Cuda == device) {
            return addCuda<rMemory, rPadding, uMemory, uPadding>((const Matrix<T, Memory, Padding> &) (*this), b);
        } else /*Host*/ {
            //return addHost<T, rMemory, rPadding>((const Matrix<T, Memory, Padding> &) (*this), b);
            return addHost<rMemory, rPadding, uMemory, uPadding>((const Matrix<T, Memory, Padding> &) (*this), b);
        }
    }

    template<class T, template<class, class> class Memory, class Padding>
    template<template<class, class> class rMemory, class rPadding,
            template<class, class> class uMemory, class uPadding>
    Matrix<T, rMemory, rPadding> Matrix<T, Memory, Padding>::addHost(const Matrix<T, Memory, Padding> &a,
                                                          const Matrix<T, uMemory, uPadding> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }

        // Since C is to be calculated on host, define as paged memory with return Padding.
        Matrix<T, PagedMemoryBlock, rPadding> c(a.getRows(), a.getCols());

        // a & b are both derived from HostMemoryBlock
        if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Padding>
                && IsDerivedFromHostMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnHost(a, b, c);
        }

        // Only a is derived from HostMemoryBlock
        else if constexpr(IsDerivedFromHostMemoryBlock<T, Memory, Padding>
                && !IsDerivedFromHostMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnHost(a, b.template copyTo<PagedMemoryBlock, uPadding>(), c);
        }

        // Only b is derived from HostMemoryBlock
        else if constexpr(!IsDerivedFromHostMemoryBlock<T, Memory, Padding>
                && IsDerivedFromHostMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnHost(a.template copyTo<PagedMemoryBlock, Padding>(), b, c);
        }

        // Neither a or b derived from HostMemoryBlock
        else {
            Internal::_addMatricesOnHost(a.template copyTo<PagedMemoryBlock, Padding>(),
                                         b.template copyTo<PagedMemoryBlock, uPadding>(), c);
        }

        // TODO: move c to requested
        return std::move(c).template moveTo<rMemory, rPadding>();
    }

    template<class T, template<class, class> class Memory, class Padding>
    template<template<class, class> class rMemory, class rPadding,
            template<class, class> class uMemory, class uPadding>
    Matrix<T, rMemory, rPadding> Matrix<T, Memory, Padding>::addCuda(const Matrix<T, Memory, Padding> &a,
                                                                         const Matrix<T, uMemory, uPadding> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }

        // Since C is to be calculated on host, define as paged memory with return Padding.
        std::unique_ptr<Matrix<T, DeviceMemoryBlock, rPadding>> aDevicePtr;
        std::unique_ptr<Matrix<T, DeviceMemoryBlock, rPadding>> bDevicePtr;


        Matrix<T, DeviceMemoryBlock, rPadding> c(a.getRows(), a.getCols());

        // a & b are both derived from HostMemoryBlock
        if constexpr(IsDerivedFromDeviceMemoryBlock<T, Memory, Padding>
                     && IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnCuda(a, b, c);
            //Internal::_addMatricesOnCuda(a, b, c);
        }

        // Only a is derived from HostMemoryBlock
        else if constexpr(IsDerivedFromDeviceMemoryBlock<T, Memory, Padding>
                          && !IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnCuda(a, b.template copyTo<DeviceMemoryBlock, uPadding>(), c);
        }

        // Only b is derived from HostMemoryBlock
        else if constexpr(!IsDerivedFromDeviceMemoryBlock<T, Memory, Padding>
                          && IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding>) {
            Internal::_addMatricesOnCuda(a.template copyTo<DeviceMemoryBlock, Padding>(), b, c);
        }

        // Neither a or b derived from HostMemoryBlock
        else if constexpr (!IsDerivedFromDeviceMemoryBlock<T, Memory, Padding> &&
                !IsDerivedFromDeviceMemoryBlock<T, uMemory, uPadding>){
            Internal::_addMatricesOnCuda(a.template copyTo<DeviceMemoryBlock, Padding>(),
                                         b.template copyTo<DeviceMemoryBlock, uPadding>(), c);
        }

        // TODO: move c to requested
        return std::move(c).template moveTo<rMemory, rPadding>();
    }
//    template<class T, template<class, class> class Memory, class Padding>
//    template<template<class, class> class uMemory, class uPadding,
//        template<class, class> class rMemory, class rPadding>
//    Matrix<T, rMemory, rPadding> Matrix<T, Memory, Padding>::addCuda(const Matrix<T, Memory, Padding> &a, const Matrix<T, uMemory, uPadding> &b) {
//        if (!a.validateMatricesAreSameShape(b)) {
//            // TODO: throw exception eventually
//            //throw a.MatrixIsInvalidShape("");
//        }
//
//        return Matrix<T, rMemory, rPadding>();
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





