# The examples are consumers of the public header, built by the same gate that
# builds the library so a change to include/cudec.h that breaks a consumer reds
# a PR instead of a user's build (issue #158).
#
# Two shapes, because the host-only configuration is toolchain-free by design:
# it compiles src/version.c alone, so cudec_lz4f_decompress is not in the
# archive and nothing that calls it can LINK there. Compiling it is still the
# half that carries the property this directory exists for, since a header a
# consumer cannot compile against is broken in both configurations. A device
# configuration - either backend, the frame example calls no runtime of its
# own - links and produces the binary CI then checks for.
if(CUDEC_DEVICE_LANGUAGE)
  add_executable(example_decode_frame decode_frame.c)
  # cudec's frame orchestration is C++, so the archive carries objects that
  # need the C++ runtime; CMake picks the link driver from the consuming
  # target's languages, and a C-only one would not get it. The same
  # consumer-side half tests/consumer/CMakeLists.txt states for the package.
  set_target_properties(example_decode_frame PROPERTIES LINKER_LANGUAGE CXX)
else()
  add_library(example_decode_frame OBJECT decode_frame.c)
endif()

# The library target, for its include directory and (in the CUDA
# configuration) its objects. PRIVATE: an example is consumed by nobody.
target_link_libraries(example_decode_frame PRIVATE cudec)

# In the sanitizer configuration the archive this links is instrumented, so an
# uninstrumented consumer of it does not link at all: the sanitizer runtime
# calls compiled into src/frame.cpp resolve nowhere. Measured, on the first CI
# run of this change - `undefined reference to __asan_report_load1` out of
# frame.cpp while linking this target. The example takes the same flags the
# tests do rather than being excluded from that configuration, because a
# consumer that is not built there is a consumer nothing checks.
if(CUDEC_SANITIZE)
  target_compile_options(
    example_decode_frame
    PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
  target_link_options(example_decode_frame PRIVATE
                      ${CUDEC_SANITIZE_LINK_FLAGS})
endif()

# The batch consumer (issue #159) has only one shape, unlike the frame one:
# it drives the device entry point and includes the CUDA runtime, so there is
# no host-only configuration of it to compile. It is also kept out of the
# sanitizer configuration for the reason the top-level CMakeLists gives for
# omitting bench/ there - an nvcc-linked binary cannot link the instrumented
# archive without dragging the sanitizer onto the device link, so the
# alternative would be to instrument the device build rather than to skip a
# consumer. It is compiled and linked, never run: the CI runner has no GPU.
if(CUDEC_ENABLE_CUDA AND NOT CUDEC_SANITIZE)
  find_package(CUDAToolkit REQUIRED)
  add_executable(example_decode_batch decode_batch.cu)
  target_link_libraries(example_decode_batch PRIVATE cudec CUDA::cudart)
  set_target_properties(
    example_decode_batch
    PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES
               "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# This directory registers no ctest entries, and it is added after the sweep
# that makes a missing TIMEOUT unforgettable (the top-level
# cudec_sweep_test_timeouts call), so a test added here would escape the sweep
# rather than be caught by it. Say so to the tree instead of in a comment: an
# add_test below this line reds the configure, and whoever writes one moves
# this directory above the sweep and deletes this block.
get_property(_example_tests DIRECTORY PROPERTY TESTS)
if(_example_tests)
  message(
    FATAL_ERROR
      "examples/ registered a ctest entry (${_example_tests}), and this "
      "directory is added after the timeout sweep that would have checked it "
      "(issue #158)")
endif()
