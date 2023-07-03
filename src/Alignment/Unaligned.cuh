//
// Created by steve on 5/26/2023.
//

#ifndef NANL_UNALIGNED_CUH
#define NANL_UNALIGNED_CUH

#include "BaseAlignment.cuh"

namespace NaNL {
    template<class T, template<typename> class Memory>
    class Unaligned : public BaseAlignment<T,Memory>{
    public:
        inline Unaligned(uint64_t rows, uint64_t cols);
        inline uint64_t getRows() const;
        inline uint64_t getActualRows() const;
        inline uint64_t getCols() const;
        inline uint64_t getActualCols() const;
        inline uint64_t getTotalSize() const;
        inline uint64_t getActualTotalSize() const;
        inline void align(uint64_t rows, uint64_t cols);
    };

} // NaNL

#include "Unaligned.cu"

#endif //NANL_UNALIGNED_CUH
