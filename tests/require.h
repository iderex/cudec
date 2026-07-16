/* The harness's entire assertion layer. ctest is the runner; each test
 * group is one executable that aborts on its first failed check (a later
 * check is meaningless once earlier state diverged). The framework
 * decision and its reassessment trigger: docs/MASTERPLAN.md section 5. */
#ifndef CUDEC_TESTS_REQUIRE_H
#define CUDEC_TESTS_REQUIRE_H

#include <cstddef>
#include <cstdio>

#define REQUIRE(cond)                                                      \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                           \
            return 1;                                                      \
        }                                                                  \
    } while (0)

/* Same, with a printf-style context suffix: the (fixture, mutant, index)
 * coordinates that make a deterministic failure directly reproducible. */
#define REQUIRE_CTX(cond, ...)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s | ", __FILE__, __LINE__,  \
                         #cond);                                           \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fputc('\n', stderr);                                      \
            return 1;                                                      \
        }                                                                  \
    } while (0)

#define REQUIRE_CUDA(call) REQUIRE((call) == cudaSuccess)

/* Byte-diff for the buffer-shaped assertion domain: reports the first
 * mismatch offset plus a small hex window instead of a bare boolean. */
inline bool equal_bytes(const void* actual_p, const void* expected_p,
                        size_t size) {
    const unsigned char* actual = static_cast<const unsigned char*>(actual_p);
    const unsigned char* expected =
        static_cast<const unsigned char*>(expected_p);
    for (size_t i = 0; i < size; i++) {
        if (actual[i] == expected[i]) {
            continue;
        }
        std::fprintf(stderr, "first byte mismatch at offset %zu:\n", i);
        const size_t from = i < 8 ? 0 : i - 8;
        for (size_t j = from; j < from + 16 && j < size; j++) {
            std::fprintf(stderr, "  [%zu] have %02x want %02x%s\n", j,
                         actual[j], expected[j], j == i ? "  <--" : "");
        }
        return false;
    }
    return true;
}

#endif /* CUDEC_TESTS_REQUIRE_H */
