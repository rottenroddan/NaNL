//
// Created by steve on 6/1/2023.
//

#ifndef NANL_MATRIXUTILITY_CUH
#define NANL_MATRIXUTILITY_CUH

#include <type_traits>


namespace NaNL {

    namespace Internal {
        class MatrixUtility {
        private:
            template<class T>
            static void addHost_Paged_Paged(const Matrix<T, PagedMemoryBlock, Unaligned> &a,
                                             const Matrix<T, PagedMemoryBlock, Unaligned> &b,
                                             Matrix<T, PagedMemoryBlock, Unaligned> &c);

            template<class T, template<typename> class Memory, template<class, template<typename> class> class Alignment,
                    template<typename> class uMemory, template<class, template<typename> class> class uAlignment>
            static void _copy(const Matrix<T, Memory, Alignment> &first, Matrix<T, uMemory, uAlignment> &second);

        public:
            template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment, template<typename> class Memory,
                    template<class, template<typename> class> class Alignment,
                    template<typename> class uMemory,
                    template<class, template<typename> class> class uAlignment>
            static Matrix<T, rMemory, rAlignment>
            addHost(const Matrix<T, Memory, Alignment> &a, const Matrix<T, uMemory, uAlignment> &b);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &a, const Matrix<T, PagedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);
//        template<class T, template<typename> class rMemory, template<class, template<typename> class> class rAlignment>
//        static Matrix<T, rMemory, rAlignment> add(const Matrix<T, PinnedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b, MatrixDeviceOperation device);

            //        template<class T>
//        static void add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b);
        };
    }
} // NaNL

#include "MatrixUtility.cu"

#endif //NANL_MATRIXUTILITY_CUH
