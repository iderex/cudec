#include "cudec.h"

int cudec_version(void) {
    return CUDEC_VERSION_MAJOR * 10000 + CUDEC_VERSION_MINOR * 100 +
           CUDEC_VERSION_PATCH;
}
