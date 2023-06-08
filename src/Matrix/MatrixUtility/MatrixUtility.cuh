//
// Created by steve on 6/1/2023.
//

#ifndef NANL_MATRIXUTILITY_CUH
#define NANL_MATRIXUTILITY_CUH




namespace NaNL {
    class MatrixUtility {
    private:
        template<class T>
        static void _addHost_Paged_Paged(const Matrix<T, PagedMemoryBlock, Unaligned> &a,
                                     const Matrix<T, PagedMemoryBlock, Unaligned> &b,
                                     Matrix<T, PagedMemoryBlock, Unaligned> &c);

    public:
        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &a, const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);

        //        template<class T>
//        static void add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b);
    };
} // NaNL

#include "MatrixUtility.cu"

#endif //NANL_MATRIXUTILITY_CUH
