//
// Created by steve on 7/17/2023.
//

#ifndef NANL_MATRIXTYPETRAITS_CUH
#define NANL_MATRIXTYPETRAITS_CUH

#include "../../MemoryBlock/PagedMemoryBlock.cuh"
#include "../../MemoryBlock/PinnedMemoryBlock.cuh"
#include "../../MemoryBlock/DeviceMemoryBlock.cuh"
#include "../Matrix.cuh"

/**
 * TODO: Debate moving these out of this file. Scared of circular dependencies here. Tightly couple includes
 *          Matrix.cuh
 *          /        \
 *         / MatrixTypeTraits.cuh
 *   Matrix.cu -------^
 */

namespace NaNL::Internal {
    template<class T, template<class, class> class Memory, class Alignment>
    constexpr bool is_matrix_derived_from_paged = std::is_base_of_v<Matrix<T, PagedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>;

    template<class T, template<class, class> class Memory, class Alignment>
    constexpr bool is_matrix_derived_from_pinned = std::is_base_of_v<Matrix<T, PinnedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>;

    template<class T, template<class, class> class Memory, class Alignment>
    constexpr bool is_matrix_derived_from_device = std::is_base_of_v<Matrix<T, DeviceMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>;

    template<class T, template<class, class> class Memory, class Alignment>
    constexpr bool is_matrix_derived_from_paged_or_pinned = std::is_base_of_v<Matrix<T, PagedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>
            || std::is_base_of_v<Matrix<T, PinnedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>;

    template<class T, template<class, class> class Memory, class Alignment>
    constexpr bool is_matrix_derived_from_pinned_or_device = std::is_base_of_v<Matrix<T, PinnedMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>
            || std::is_base_of_v<Matrix<T, DeviceMemoryBlock, Alignment>, Matrix<T, Memory, Alignment>>;
}


#endif //NANL_MATRIXTYPETRAITS_CUH
