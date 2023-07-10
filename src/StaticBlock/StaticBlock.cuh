/*#pragma once
#ifndef NANL_STATICBLOCK_CUH
#define NANL_STATICBLOCK_CUH

#define CONCATE_(X,Y) X##Y
#define CONCATE(X,Y) CONCATE_(X,Y)
#define UNIQUE(NAME) CONCATE(NAME, __LINE__)

struct Static_
{
    template<typename T> Static_ (T only_once) { only_once(); }
    ~Static_ () {}  // to counter "warning: unused variable"
};
// `UNIQUE` macro required if we expect multiple `static` blocks in function
#define _static_block_ static Static_ UNIQUE(block) = []() -> void

#endif //NANL_STATICBLOCK_CUH*/
