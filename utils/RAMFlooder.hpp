#ifndef __RAM_FLOODER_HPP__
#define __RAM_FLOODER_HPP__

#include <array>
#include <cstddef>
#include "MemoryDefs.hpp"

#ifndef RAM_FLOODER_ON_CORE
// This must be a bare integer token: MEML_RUNS_ON_CORE stringizes it into a linker section name.
#define RAM_FLOODER_ON_CORE    1
#endif


namespace utils {

template <typename T, std::size_t Size>
class RAMFlooder {

public:

    RAMFlooder() : m_data{} {}

    MEML_RUNS_ON_CORE(RAM_FLOODER_ON_CORE) void FillOnce() {
        // Make volatile to prevent optimisation
        volatile T* p = m_data.data();
        std::size_t sizeUnrolled = Size / 8 * 8;
        for (std::size_t i = 0; i < sizeUnrolled; i += 8) {
            p[i + 0] = T(i); p[i + 1] = T(i); p[i + 2] = T(i); p[i + 3] = T(i);
            p[i + 4] = T(i); p[i + 5] = T(i); p[i + 6] = T(i); p[i + 7] = T(i);
        }
        for (std::size_t i = sizeUnrolled; i < Size; ++i) p[i] = T(i);
    }

private:
    std::array<T, Size> m_data;

};

};  // namespace utils

#endif  // __RAM_FLOODER_HPP__
