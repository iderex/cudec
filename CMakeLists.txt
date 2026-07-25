cmake_minimum_required(VERSION 3.24)

project(cudec VERSION 0.0.1 LANGUAGES C)

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

add_library(cudec src/version.c)
target_include_directories(cudec PUBLIC include)

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
