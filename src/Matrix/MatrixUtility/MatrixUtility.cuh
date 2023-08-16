//
// Created by steve on 6/1/2023.
//

#ifndef NANL_MATRIXUTILITY_CUH
#define NANL_MATRIXUTILITY_CUH

#include <type_traits>


namespace NaNL::Internal {

        class MatrixUtility {
        private:


//            template<class T, template<class, class> class Memory, class Alignment,
//                    template<class, class> class uMemory, class uAlignment,
//                    template<class, class> class rMemory, class rAlignment>
//            static void _addCuda(const Matrix<T, Memory, Alignment> &a,
//                                            const Matrix<T, uMemory, uAlignment> &b,
//                                            Matrix<T, rMemory, rAlignment> &c)
//            requires IsDerivedFromDeviceMemoryBlock<T, Memory, Alignment> &&
//                    IsDerivedFromDeviceMemoryBlock<T, uMemory, uAlignment> &&
//                    IsDerivedFromDeviceMemoryBlock<T, rMemory, rAlignment>;

        public:





            //        template<class T>
//        static void add(const Matrix<T, PagedMemoryBlock, Unaligned> &a, const Matrix<T, PinnedMemoryBlock, Unaligned> &b);
        };
    } // NaNL

#include "MatrixUtility.cu"

#endif //NANL_MATRIXUTILITY_CUH
