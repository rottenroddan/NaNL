//
// Created by steve on 5/26/2023.
//

#ifndef NANL_UNALIGNED_CUH
#define NANL_UNALIGNED_CUH

#include "BaseAlignment.cuh"

namespace NaNL {
    class Unaligned : public BaseAlignment {
    public:
        inline Unaligned(uint64_t rows, uint64_t cols);
    };

    template<typename Alignment>
    concept IsAlignmentTypeDerivedFromUnaligned = std::derived_from<Alignment, Unaligned>;

} // NaNL

#include "Unaligned.cu"

#endif //NANL_UNALIGNED_CUH
