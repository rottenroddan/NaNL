//
// Created by steve on 6/1/2023.
//

#ifndef NANL_MATRIXUTILITY_CUH
#define NANL_MATRIXUTILITY_CUH

#include <type_traits>


namespace NaNL::Internal {
        template<typename T, template<typename, typename> class U, class V>
        concept IsDerivedFromHostMemoryBlock = std::is_base_of_v<NaNL::Internal::HostMemoryBlock<T, V>, U<T,V>>;

        class MatrixUtility {
        private:
            template<class T, template<class, class> class Memory, class Alignment,
                    template<class, class> class uMemory, class uAlignment,
                    template<class, class> class rMemory, class rAlignment>
            static void addHost_Paged_Paged(const Matrix<T, Memory, Alignment> &a,
                                            const Matrix<T, uMemory, uAlignment> &b,
                                            Matrix<T, rMemory, rAlignment> &c)
                                            requires IsDerivedFromHostMemoryBlock<T, Memory, Alignment> &&
                                                    IsDerivedFromHostMemoryBlock<T, uMemory, uAlignment> &&
                                                    IsDerivedFromHostMemoryBlock<T, rMemory, rAlignment>;

            template<class T, template<class, class> class Memory, class Alignment,
                    template<class, class> class uMemory, class uAlignment>
            static void _copy(const Matrix<T, Memory, Alignment> &first, Matrix<T, uMemory, uAlignment> &second);

        public:
            template<class T, template<class, class> class rMemory, class rAlignment,
                    template<class, class> class Memory, class Alignment,
                    template<class, class> class uMemory, class uAlignment>
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
    } // NaNL

#include "MatrixUtility.cu"

#endif //NANL_MATRIXUTILITY_CUH
