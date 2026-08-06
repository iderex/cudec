# The examples are consumers of the public header, built by the same gate that
# builds the library so a change to include/cudec.h that breaks a consumer reds
# a PR instead of a user's build (issue #158).
#
# Two shapes, because the host-only configuration is toolchain-free by design:
# it compiles src/version.c alone, so cudec_lz4f_decompress is not in the
# archive and nothing that calls it can LINK there. Compiling it is still the
# half that carries the property this directory exists for, since a header a
# consumer cannot compile against is broken in both configurations. The CUDA
# configuration links and produces the binary CI then checks for.
if(CUDEC_ENABLE_CUDA)
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
