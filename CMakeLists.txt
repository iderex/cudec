cmake_minimum_required(VERSION 3.24)

# The version has ONE source: the CUDEC_VERSION_* macros in include/cudec.h.
# CMake reads them instead of restating the number, so a release bump touches
# one file and package metadata cannot drift from what cudec_version()
# reports. Reading the header (rather than configuring one) keeps it
# self-contained for consumers who never run CMake. Fail-closed: a macro that
# is missing, renamed, or not a plain integer is a configure error, never a
# silent empty version - and the derivation is checked against the header a
# second time at runtime, in tests/conformance_c.c.
file(READ "${CMAKE_CURRENT_LIST_DIR}/include/cudec.h" _cudec_public_header)
foreach(_component MAJOR MINOR PATCH)
  if(NOT _cudec_public_header MATCHES
     "\n#define CUDEC_VERSION_${_component} ([0-9]+)")
    message(
      FATAL_ERROR
        "include/cudec.h defines no integer CUDEC_VERSION_${_component}; "
        "the project version is read from those three macros.")
  endif()
  set(_cudec_version_${_component} "${CMAKE_MATCH_1}")
endforeach()
unset(_cudec_public_header)

project(
  cudec
  VERSION
    "${_cudec_version_MAJOR}.${_cudec_version_MINOR}.${_cudec_version_PATCH}"
  LANGUAGES C)

# CMAKE_INSTALL_INCLUDEDIR / _LIBDIR are read below by the install interface of
# the library target, so this has to come before the target is defined rather
# than beside the install() rules at the end.
include(GNUInstallDirs)

# The CUDA engine is opt-in and fail-closed: asking for it without a CUDA
# toolchain is a hard configure error, never a silent host-only fallback (a
# CI job that passed without compiling the kernels would be a fail-open
# gate). CI flips this on together with the CUDA toolchain (issue #3).
option(CUDEC_ENABLE_CUDA "Build the CUDA decode engine and its tests" OFF)

# Host-side sanitizer gate (issue #42): opt-in, default OFF. When set (e.g.
# -DCUDEC_SANITIZE=address,undefined) it instruments the HOST untrusted-input
# attack surface - the frame parse (src/frame.cpp), the single-sourced block
# parser exercised by the CPU twin, and the C-ABI argument validation - and the
# GPU-less test subset runs under the sanitizer, so an out-of-bounds access, an
# uninitialised read, or integer/shift UB on a hostile bitstream reds the
# build. It never touches device (nvcc) compilation: host ASan cannot
# instrument device code, and the device net stays compute-sanitizer plus the
# whole-corpus GPU/oracle byte diff (masterplan section 5). Fail-closed:
# -fno-sanitize-recover=all makes a UBSan finding abort with a non-zero status
# instead of printing and continuing green.
set(CUDEC_SANITIZE ""
    CACHE STRING
          "Host sanitizers to enable, e.g. 'address,undefined' (empty = off)")
if(CUDEC_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "CUDEC_SANITIZE requires a GCC/Clang host toolchain "
                        "(the pinned Linux container is the maintained path).")
  endif()
  # Host C/C++ only - the generator expression keeps these off any CUDA
  # translation unit. The compile flags instrument the TUs; the link flags pull
  # in the sanitizer runtimes on the executables that run under the gate.
  set(CUDEC_SANITIZE_COMPILE_FLAGS
      -fsanitize=${CUDEC_SANITIZE} -fno-sanitize-recover=all
      -fno-omit-frame-pointer)
  set(CUDEC_SANITIZE_LINK_FLAGS -fsanitize=${CUDEC_SANITIZE})
endif()

# Static only, and pinned rather than left to a global switch (issue #79). A
# shared cudec needs SOVERSION discipline and symbol-visibility work - decide
# what is exported, hide the rest, hold the soname to the ABI - and none of that
# exists yet. Building one anyway would produce a library whose ABI story is
# unmanaged while looking exactly like one whose story is managed, which is the
# worse of the two failures. Revisit when a packager asks for it or at 1.0,
# whichever comes first; the work is the visibility audit, not this line.
#
# The explicit STATIC is what enforces it, so BUILD_SHARED_LIBS=ON cannot
# produce a shared cudec by accident. The refusal below is scoped to a top-level
# build on purpose: a superbuild that sets BUILD_SHARED_LIBS globally and
# vendors cudec is asking for shared *everything else*, and killing its
# configure over a library it gets statically either way would be this project
# reaching outside its own build.
add_library(cudec STATIC src/version.c)
if(BUILD_SHARED_LIBS AND PROJECT_IS_TOP_LEVEL)
  message(
    FATAL_ERROR
      "cudec builds as a static library only. BUILD_SHARED_LIBS=ON would need "
      "an SOVERSION and a symbol-visibility policy this project has not "
      "settled (issue #79).")
endif()

# The in-tree spelling of the exported name, so a consumer inside this build
# and a consumer of the installed package link the same string.
add_library(cudec::cudec ALIAS cudec)
# One include directory, two spellings, because the path differs between the
# tree it is built in and the prefix it is installed to. The install-tree half
# is stated here and NOT a second time via install(TARGETS ... INCLUDES
# DESTINATION): both write the same entry, and using both ships the exported
# INTERFACE_INCLUDE_DIRECTORIES with the directory listed twice.
target_include_directories(
  cudec PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(NOT MSVC)
  target_compile_options(cudec
                         PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror>)
endif()

# Instrument the library's host TUs (version.c now; frame.cpp/stream.cpp once
# the CUDA engine is enabled). Guarded to C/C++ so batch.cu is never sanitized.
if(CUDEC_SANITIZE)
  target_compile_options(
    cudec
    PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
endif()

if(CUDEC_ENABLE_CUDA)
  if(MSVC)
    # cl.exe ignores the GCC-style strict-warning flags below with a
    # warning, not an error - the warnings gate would silently drop.
    message(
      FATAL_ERROR
        "CUDEC_ENABLE_CUDA requires a GCC/Clang host toolchain (the pinned "
        "Linux container is the maintained path); MSVC-host CUDA builds are "
        "unsupported.")
  endif()

  # sm_80 is the baseline the kernels are written against; sm_86 is the
  # tuned primary target (RTX 3080). SASS for both, PTX only for the
  # baseline - sm_80 PTX already JIT-covers every newer device, so a second
  # PTX copy would be dead weight in the fatbin. Overridable from the
  # command line.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86-real)
  endif()
  enable_language(CUDA)
  # The project language is C; the test net and the bench harness add .cpp
  # translation units, so CXX is enabled here where both subdirectories
  # can see it (a language enabled inside one subdirectory is invisible to
  # its siblings).
  enable_language(CXX)
  enable_testing()

  # Every ctest entry carries a finite TIMEOUT (issue #72). A decoder that
  # fails to terminate hangs the runner instead of failing it, and a hang is
  # a fail-closed violation exactly like an out-of-bounds read - so the
  # timeout is part of the gate, not a convenience. The value is generous
  # against a loaded machine and a sanitized (-O0 + ASan) build, and finite
  # against "never".
  set(CUDEC_TEST_TIMEOUT_SECONDS 300)
  # An upper bound too: a TIMEOUT of 99999999 is finite and useless, and the
  # point of the property is that a hang reds the run inside a human wait.
  set(CUDEC_TEST_TIMEOUT_CEILING 3600)

  # ...and the property must not be forgettable. A test property can only be
  # read from the directory that declared it (get_test_property is
  # directory-scoped before CMake 3.28, and the pinned minimum here is 3.24),
  # so this runs in two halves: each directory that registers tests calls
  # cudec_assert_test_timeouts(), which checks and RECORDS what it verified,
  # and the sweep below - run once, last, from the top level - walks every
  # subdirectory and fails on any registered test that was never recorded.
  # That closes both ways the per-directory call is forgettable: a future
  # add_subdirectory that never calls it, and an add_test appended after the
  # call site.
  function(cudec_assert_test_timeouts)
    get_property(_tests DIRECTORY PROPERTY TESTS)
    foreach(_test IN LISTS _tests)
      get_test_property(${_test} TIMEOUT _timeout)
      if(NOT _timeout OR NOT _timeout MATCHES "^[0-9]+$"
         OR _timeout GREATER ${CUDEC_TEST_TIMEOUT_CEILING})
        message(
          FATAL_ERROR
            "ctest entry '${_test}' has no usable TIMEOUT (got '${_timeout}', "
            "want 1..${CUDEC_TEST_TIMEOUT_CEILING} seconds): every test must "
            "fail on a hang rather than stall the run (issue #72)")
      endif()
      # Keyed by directory too: two directories may register the same test
      # name, and a bare-name key would let the unchecked one ride on the
      # checked one.
      set_property(GLOBAL APPEND PROPERTY CUDEC_TIMEOUT_CHECKED_TESTS
                   "${CMAKE_CURRENT_SOURCE_DIR}::${_test}")
    endforeach()
  endfunction()

  function(cudec_sweep_test_timeouts dir)
    get_property(_checked GLOBAL PROPERTY CUDEC_TIMEOUT_CHECKED_TESTS)
    get_property(_tests DIRECTORY ${dir} PROPERTY TESTS)
    foreach(_test IN LISTS _tests)
      if(NOT "${dir}::${_test}" IN_LIST _checked)
        message(
          FATAL_ERROR
            "ctest entry '${_test}' (registered in ${dir}) was never checked "
            "for a TIMEOUT: call cudec_assert_test_timeouts() at the END of "
            "that directory's CMakeLists.txt (issue #72)")
      endif()
    endforeach()
    get_property(_subs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(_sub IN LISTS _subs)
      cudec_sweep_test_timeouts(${_sub})
    endforeach()
  endfunction()

  # Single source of truth for the strict device flags: the raw list also
  # feeds the configure-time liveness probes in tests/CMakeLists.txt, so an
  # edit here is probed, not trusted.
  set(CUDEC_CUDA_STRICT_FLAGS_RAW --Werror=all-warnings
                                  -Xcompiler=-Wall,-Wextra,-Werror)
  set(CUDEC_CUDA_STRICT_FLAGS
      $<$<COMPILE_LANGUAGE:CUDA>:${CUDEC_CUDA_STRICT_FLAGS_RAW}>)

  # frame.cpp (the LZ4 frame orchestrator) is host C++ that drives the
  # device batch decoder, so it needs the CUDA runtime.
  find_package(CUDAToolkit REQUIRED)
  target_sources(cudec PRIVATE src/batch.cu src/frame.cpp src/stream.cpp)
  target_link_libraries(cudec PRIVATE CUDA::cudart)
  # The cudec target predates enable_language(CUDA), so the architecture
  # list must be applied as an explicit target property.
  set_target_properties(
    cudec PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                     CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                     CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    cudec PRIVATE ${CUDEC_CUDA_STRICT_FLAGS}
                  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)

  # Keep tests/ after everything that may touch the cudec target: its
  # conformance checks snapshot the target's properties at this point -
  # anything added to the target below this line would escape them. bench/
  # follows because it consumes targets defined in tests/ and is
  # consumer-only by rule (it links project targets, never modifies them).
  add_subdirectory(tests)
  # bench_lz4 is a single nvcc-linked binary (bench_lz4.cpp + gpu_bench.cu); it
  # cannot link the ASan-instrumented cudec without dragging the sanitizer onto
  # the device link, so the host-only sanitizer gate omits bench entirely. Its
  # CPU self-checks stay covered by the normal (non-sanitized) build.
  if(NOT CUDEC_SANITIZE)
    add_subdirectory(bench)
  endif()

  # Last, so it sees every directory both subdirectories added: no ctest entry
  # anywhere in the tree may have escaped the timeout check (issue #72).
  cudec_sweep_test_timeouts(${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---- The install surface (issue #79). A release nobody can install is not a
# release, and until this existed every consumer had to vendor the tree.
#
# Placed after both configurations so the exported target is whatever this
# configure actually built, and after tests/ so nothing here can slip a
# property past the conformance snapshot inside it: install() rules read the
# target, they do not modify it.
#
# Top-level only. A project that vendors cudec by add_subdirectory or
# FetchContent gets the target, not the packaging: without this guard its own
# `cmake --install` would ship cudec's header, archive and package config into
# the parent's prefix, where find_package(cudec) would then find a config the
# parent never meant to publish and shadow a real install.
if(PROJECT_IS_TOP_LEVEL)
  include(CMakePackageConfigHelpers)

  # No INCLUDES DESTINATION here: the install-tree include path is already in
  # the target's interface as $<INSTALL_INTERFACE:...>, and stating it both
  # ways writes the same directory into the exported
  # INTERFACE_INCLUDE_DIRECTORIES twice.
  install(
    TARGETS cudec
    EXPORT cudec-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # The whole public surface is one header, by design: include/cudec.h needs no
  # CUDA headers and pulls in nothing but stddef.h and stdint.h. src/ is
  # internal and is not installed.
  install(FILES include/cudec.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(
    EXPORT cudec-targets
    FILE cudec-targets.cmake
    NAMESPACE cudec::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudec)

  # SameMinorVersion, and the reason is the 0.x in the version rather than a
  # preference. SemVer itself promises nothing at 0.y.z - "anything MAY change
  # at any time" - so this is a convention this project adopts and states,
  # not a rule read out of the specification. The convention is the common one:
  # at 0.x the minor bump is the breaking one, so 0.0.1 and 0.1.0 are not
  # interchangeable and a consumer asking for 0.0 must not be handed 0.1.
  # ExactVersion was the alternative and is too strict in the other direction:
  # it would refuse 0.0.2 to a consumer who wrote 0.0.1 and meant the patch
  # series, which is the axis this project intends to keep compatible.
  #
  # The revisit is at 1.0, where the same convention gives SameMajorVersion.
  # That is a deliberate change of policy at a deliberate moment, not a default
  # to drift into.
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cudec-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

  configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/cudec-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cudec-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudec)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cudec-config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/cudec-config-version.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudec)
endif()
