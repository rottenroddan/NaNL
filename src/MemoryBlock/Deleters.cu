//
// Created by steve on 5/26/2023.
//

#ifndef NANL_DELETERS_CU
#define NANL_DELETERS_CU

namespace NaNL {
    /**
     * Function that frees paged memory from the pointer provided.
     * @tparam T Pointer type.
     * @param ptr_ Pointer to the address to free.
     */
    template<typename T>
    inline void _freePagedMemory(T *ptr_) { delete[] ptr_; }

    /**
     * Function that frees pinned memory from the pointer provided.
     * Calls <code>cudaFreeHost</code> to free pinned memory.
     * @tparam T Pointer type.
     * @param ptr_ Pointer to the address to free.
     */
    template<typename T>
    inline void _freePinnedMemory(T *ptr_) { cudaFreeHost(ptr_); }
}

#endif