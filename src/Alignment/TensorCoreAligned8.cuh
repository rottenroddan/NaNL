//
// Created by steve on 7/31/2023.
//

#ifndef NANL_TENSORCOREALIGNED8_CUH
#define NANL_TENSORCOREALIGNED8_CUH

#include "BaseAlignment.cuh"
#include "TensorCoreAligned128Bits.cuh"
namespace NaNL {
    class TensorCoreAligned8 : public Internal::TensorCoreAligned128Bits {
    protected:
        static constexpr uint64_t TYPE_SIZE_BITS = 8;
        static constexpr uint64_t ALIGNMENT_MULTIPLE_OF = TOTAL_TENSOR_BIT_ALIGNMENT / TYPE_SIZE_BITS;
    public:
        inline TensorCoreAligned8(uint64_t rows, uint64_t cols);
    };
}

#include "TensorCoreAligned8.cu"

#endif //NANL_TENSORCOREALIGNED8_CUH
