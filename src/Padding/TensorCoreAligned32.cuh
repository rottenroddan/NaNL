//
// Created by steve on 8/1/2023.
//

#ifndef NANL_TENSORCOREALIGNED32_CUH
#define NANL_TENSORCOREALIGNED32_CUH

#include "BaseAlignment.cuh"
#include "TensorCoreAligned128Bits.cuh"
namespace NaNL {
    class TensorCoreAligned32 : public Internal::TensorCoreAligned128Bits {
    protected:
        static constexpr uint64_t TYPE_SIZE_BITS = 32;
        static constexpr uint64_t ALIGNMENT_MULTIPLE_OF = TOTAL_TENSOR_BIT_ALIGNMENT / TYPE_SIZE_BITS;
    public:
        inline TensorCoreAligned32(uint64_t rows, uint64_t cols);
    };
}

#include "TensorCoreAligned32.cu"

#endif //NANL_TENSORCOREALIGNED32_CUH
