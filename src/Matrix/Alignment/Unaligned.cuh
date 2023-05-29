//
// Created by steve on 5/26/2023.
//

#ifndef NANL_UNALIGNED_CUH
#define NANL_UNALIGNED_CUH

#include "BaseAlignment.cuh"

namespace NaNL {
    template<class T, template<typename> class Memory>
    class Unaligned : protected BaseAlignment<T,Memory>{
    public:
        Unaligned(uint64_t rows, uint64_t cols);
        inline uint64_t getRows();
        inline uint64_t getActualRows();
        inline uint64_t getCols();
        inline uint64_t getActualCols();
        inline uint64_t getTotalSize();
        inline uint64_t getActualTotalSize();
    };

} // NaNL

#include "Unaligned.cu"

#endif //NANL_UNALIGNED_CUH
