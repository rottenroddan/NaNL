//
// Created by steve on 5/26/2023.
//

#ifndef NANL_BASEALIGNMENT_CUH
#define NANL_BASEALIGNMENT_CUH

namespace NaNL {
    class BaseAlignment {
    protected:
        inline BaseAlignment(uint64_t rows, uint64_t cols, uint64_t multiple = 1);

        uint64_t rows{}, actualRows{};
        uint64_t cols{}, actualCols{};
        uint64_t totalSize{}, actualTotalSize{};
    public:
        [[nodiscard]] inline uint64_t getRows() const;
        [[nodiscard]] inline uint64_t getCols() const;
        [[nodiscard]] inline uint64_t getTotalSize() const;
        [[nodiscard]] inline uint64_t getActualRows() const;
        [[nodiscard]] inline uint64_t getActualCols() const;
        [[nodiscard]] inline uint64_t getActualTotalSize() const;

        inline void align(uint64_t rows, uint64_t cols, uint64_t multiple = 1);
    };

    template<typename Alignment>
    concept IsAlignmentTypeDerivedOrSimilarToBaseAlignment = std::derived_from<Alignment, BaseAlignment> ||
            requires (Alignment alignment) {
        {alignment.getRows()} -> std::convertible_to<uint64_t>;
        {alignment.getActualRows()} -> std::convertible_to<uint64_t>;
        {alignment.getCols()} -> std::convertible_to<uint64_t>;
        {alignment.getActualCols()} -> std::convertible_to<uint64_t>;
        {alignment.getTotalSize()} -> std::convertible_to<uint64_t>;
        {alignment.getActualTotalSize()} -> std::convertible_to<uint64_t>;
        {alignment.align(0, 0)} -> std::convertible_to<uint64_t>;
    } ;
}



#include "BaseAlignment.cu"

#endif //NANL_BASEALIGNMENT_CUH
