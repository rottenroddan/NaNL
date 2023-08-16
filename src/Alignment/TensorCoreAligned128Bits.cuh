//
// Created by steve on 8/1/2023.
//

#ifndef NANL_TENSORCOREALIGNED128BITS_CUH
#define NANL_TENSORCOREALIGNED128BITS_CUH

#include "BaseAlignment.cuh"
namespace NaNL::Internal {
    class TensorCoreAligned128Bits : public BaseAlignment {
    protected:
        static constexpr uint64_t TOTAL_TENSOR_BIT_ALIGNMENT = 128;
    public:
        inline TensorCoreAligned128Bits(uint64_t rows, uint64_t cols);
        inline void align(uint64_t rows, uint64_t cols, uint64_t multiple);
    };
}

#include "TensorCoreAligned128Bits.cu"
#endif //NANL_TENSORCOREALIGNED128BITS_CUH
