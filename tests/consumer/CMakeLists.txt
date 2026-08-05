# The out-of-tree consumer that proves the install surface (issue #79).
#
# Deliberately NOT added by the parent build: it is configured on its own,
# against an installed prefix, by the CI step that installs one. A consumer
# that could see the source tree would prove nothing about the package, since
# every path it needs would already be in scope.
cmake_minimum_required(VERSION 3.24)

project(cudec_consumer LANGUAGES C)

# The version this consumer asks for. Parameterised because the version file is
# otherwise dead weight in the proof: config mode only reads
# cudec-config-version.cmake when the caller requests a version, so a consumer
# that requests none would configure happily against a prefix whose version
# file was missing, or generated with the wrong compatibility mode. The CI step
# drives this twice - once with a version the policy must accept and once with
# one it must refuse - so the SameMinorVersion decision is exercised in both
# directions rather than merely installed.
set(CUDEC_CONSUMER_REQUEST_VERSION ""
    CACHE STRING "Version to pass to find_package(cudec) (empty = none)")

# REQUIRED, so a prefix that is missing the config reds the configure rather
# than falling through to a link error with a worse message. With a version
# requested it also reds on an incompatible one; with none, it does not, which
# is why the CI step always passes one.
find_package(cudec ${CUDEC_CONSUMER_REQUEST_VERSION} REQUIRED)

add_executable(consumer main.c)

# The namespaced name only, which is what an outside consumer would write. The
# bare `cudec` would resolve against a stray system library of that name; the
# namespaced one is an error if the package did not define it.
target_link_libraries(consumer PRIVATE cudec::cudec)
