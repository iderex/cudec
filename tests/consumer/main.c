/* The whole point of the consumer: an outside translation unit that knows
 * only <cudec.h> and the exported cudec::cudec target compiles, links, and
 * calls into the installed library.
 *
 * It runs as well as builds, on a machine with no GPU, because cudec_version()
 * touches no device. A compile-and-link-only check would pass against an
 * archive whose symbol was never actually reachable at run time.
 *
 * The header is included by its installed spelling. If the install put it
 * anywhere the exported include interface does not point at, this line fails
 * before anything else can. */
#include <cudec.h>

#include <stdio.h>

int main(void) {
    const int expected = CUDEC_VERSION_MAJOR * 10000 +
                         CUDEC_VERSION_MINOR * 100 + CUDEC_VERSION_PATCH;
    const int reported = cudec_version();

    /* The installed header and the installed archive are two separate files
     * in the prefix, and nothing in the package config can tell that they came
     * from one build: a config-version file compares the version the CONSUMER
     * asked for against the version the prefix declares, and has no view
     * inside the prefix at all. So this comparison is the only place that skew
     * is visible from outside, and it is cheap. */
    if (reported != expected) {
        (void)fprintf(stderr,
                      "installed cudec_version() reports %d, the installed "
                      "header says %d\n",
                      reported, expected);
        return 1;
    }

    (void)printf("cudec %d.%d.%d, consumed via find_package(cudec)\n",
                 CUDEC_VERSION_MAJOR, CUDEC_VERSION_MINOR,
                 CUDEC_VERSION_PATCH);
    return 0;
}
