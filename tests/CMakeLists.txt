# The test net: ctest is the runner, one executable per test group, the
# assertion header tests/require.h as the only "framework". Decision record
# and reassessment trigger: docs/MASTERPLAN.md section 5.

# Supply chain: the hash-verified tarball is the only acceptable oracle
# source - a system copy must never silently substitute for it.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

# liblz4, pinned by the SHA-256 of the maintainer-uploaded release asset
# (self-computed at pin time and cross-checked against conan-center; the
# auto-generated /archive/ tarballs are avoided - GitHub regenerated them
# in 2023 and their hashes moved). Test-only dependency: the link
# allowlist below keeps it out of the library.
include(FetchContent)
FetchContent_Declare(
  lz4
  URL https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
  URL_HASH
    SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(lz4)

# The uploaded tarball has no top-level CMakeLists (lz4's lives in
# build/cmake), so MakeAvailable only populates the source tree; we compile
# the one translation unit the block-format oracle needs and depend on
# nothing of lz4's build system. SYSTEM include + none of our strict flags:
# third-party code is audited by hash, not reformatted.
add_library(lz4_oracle STATIC ${lz4_SOURCE_DIR}/lib/lz4.c)
target_include_directories(lz4_oracle SYSTEM PUBLIC ${lz4_SOURCE_DIR}/lib)

# The frame oracle adds liblz4's frame API and xxHash (the checksum the
# frame format uses) - the reference the cudec frame decoder is held to.
add_library(
  lz4_frame_oracle STATIC
  ${lz4_SOURCE_DIR}/lib/lz4.c ${lz4_SOURCE_DIR}/lib/lz4hc.c
  ${lz4_SOURCE_DIR}/lib/lz4frame.c ${lz4_SOURCE_DIR}/lib/xxhash.c)
target_include_directories(lz4_frame_oracle SYSTEM PUBLIC
                           ${lz4_SOURCE_DIR}/lib)
# -O2 unconditionally: the oracle is also the bench harness's timed
# decoder, and an empty CMAKE_BUILD_TYPE (the documented container
# command) would otherwise time -O0 reference code. Correctness is
# optimization-independent; honest numbers are not.
target_compile_options(lz4_oracle PRIVATE -O2)

# PRIVATE on purpose: the archive still links through, but the lz4 include
# path stays out of every consumer's command line - so an accidental
# #include <lz4.h> in a .cu test is a compile error, and the "nvcc never
# sees lz4.h" rule is enforced, not just observed.
add_library(cudec_test_fixtures STATIC fixtures.cpp)
target_link_libraries(cudec_test_fixtures PRIVATE lz4_oracle)
target_include_directories(cudec_test_fixtures PUBLIC .)
set_target_properties(cudec_test_fixtures PROPERTIES CXX_STANDARD 17
                                                     CXX_STANDARD_REQUIRED ON)
# -O2 for the harness's UNTIMED bulk work: corpus prep and the byte-verify
# pass run through this code over 200+ MB corpora (the timed region itself
# calls LZ4_decompress_safe in lz4_oracle directly and never enters
# fixtures code). Keeps setup time negligible under the build-type-less
# documented container command.
target_compile_options(
  cudec_test_fixtures
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-O2>)
# Sanitizer gate (issue #42): the harness that builds and byte-verifies the
# corpora is host code on the untrusted-input path, so it runs under the
# sanitizer too. The lz4 oracle it links is deliberately left uninstrumented
# (third-party, audited by hash - the reference must not red the gate).
if(CUDEC_SANITIZE)
  target_compile_options(
    cudec_test_fixtures
    PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
endif()

# GPU tests carry the label (the GPU-less CI runner selects -LE gpu) and
# run serially: one physical device, and deterministic device state is
# worth more than parallel wall time. A GPU test never self-skips - the
# banned pattern is recorded in docs/MASTERPLAN.md section 5.
function(cudec_add_test name)
  cmake_parse_arguments(T "GPU" "" "SOURCES;LIBS" ${ARGN})
  # The host-only sanitizer gate (issue #42) builds only the GPU-less,
  # host-instrumentable subset: a GPU test is a .cu binary linked by nvcc,
  # which cannot pull in the ASan-instrumented cudec without sanitizing the
  # device link. Skip them so the sanitizer build stays pure host.
  if(T_GPU AND CUDEC_SANITIZE)
    return()
  endif()
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE cudec ${T_LIBS})
  set_target_properties(
    ${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                       CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                       CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    ${name} PRIVATE ${CUDEC_CUDA_STRICT_FLAGS}
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Werror>)
  # Sanitizer flags on the host test's own TUs (C/C++ only) and its link, so
  # the instrumented cudec/fixtures objects pull in the runtimes. GPU tests
  # returned above, so this only ever reaches host executables.
  if(CUDEC_SANITIZE)
    target_compile_options(
      ${name} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDEC_SANITIZE_COMPILE_FLAGS}>)
    target_link_options(${name} PRIVATE ${CUDEC_SANITIZE_LINK_FLAGS})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT
                                          ${CUDEC_TEST_TIMEOUT_SECONDS})
  if(T_GPU)
    set_tests_properties(${name} PROPERTIES LABELS gpu RUN_SERIAL TRUE)
  endif()
endfunction()

cudec_add_test(conformance_c SOURCES conformance_c.c)
set_target_properties(conformance_c PROPERTIES C_STANDARD 99
                                               C_STANDARD_REQUIRED ON)
cudec_add_test(oracle_lz4 SOURCES oracle_lz4.cpp LIBS cudec_test_fixtures)
cudec_add_test(parser_twin SOURCES parser_twin.cpp LIBS cudec_test_fixtures)
# The twin compiles the decoder core from src/ on the host - the
# fail-closed ladder runs on the GPU-less CI runner (masterplan section 9,
# ladder step 2).
target_include_directories(parser_twin
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# The frame parse's reject branches and its host-only assembly path exercised
# with NO GPU: every case rejects (or decodes an all-uncompressed frame) before
# the first CUDA call, so it runs on the GPU-less runner and, under the
# sanitizer gate (issue #42), gives src/frame.cpp's hostile-input pointer
# arithmetic ASan/UBSan coverage that the GPU-labeled frame_twin cannot get in
# CI. It computes header checksums with the library's own xxhash32.h.
cudec_add_test(frame_host_negative SOURCES frame_host_negative.cpp)
target_include_directories(frame_host_negative
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
# Termination as a tested invariant (issue #72): the parser's liveness
# contract asserted per call over the truncation ladder, the mutant corpus,
# the crafted hostile corpus, and the frame block-table walk - all of it
# rejecting (or assembling all-uncompressed) before the first CUDA call, so
# it runs on the GPU-less runner and under the sanitizer gate.
cudec_add_test(termination SOURCES termination.cpp LIBS cudec_test_fixtures)
target_include_directories(termination
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
cudec_add_test(launch_fail SOURCES launch_fail.cpp)
# Belt and suspenders with the setenv inside launch_fail.cpp: zero visible
# devices is the test condition, on the GPU machine too.
set_tests_properties(launch_fail PROPERTIES ENVIRONMENT
                                            "CUDA_VISIBLE_DEVICES=")
cudec_add_test(smoke SOURCES smoke.cu GPU)
cudec_add_test(gpu_fixture SOURCES gpu_fixture.cu LIBS cudec_test_fixtures
               GPU)
cudec_add_test(frame_twin SOURCES frame_twin.cu LIBS lz4_frame_oracle GPU)
# frame_twin verifies the library's own xxhash32.h against liblz4's XXH32.
# Guarded: the host-only sanitizer build (issue #42) skips the GPU targets, so
# this follow-up only runs when the target actually exists.
if(TARGET frame_twin)
  target_include_directories(frame_twin
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
cudec_add_test(stream_twin SOURCES stream_twin.cu LIBS cudec_test_fixtures GPU)
# The same hostile corpus on the device, behind a host-side watchdog: a warp
# that never leaves its sequence loop holds the launch, so this one polls an
# event against a deadline instead of blocking on the stream (issue #72).
cudec_add_test(termination_gpu SOURCES termination_gpu.cu LIBS
               cudec_test_fixtures GPU)
# It also launches the kernel directly, to reach a geometry the shipped entry
# point never produces (fewer threads than one warp), so it needs the internal
# header from src/.
if(TARGET termination_gpu)
  target_include_directories(termination_gpu
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
# Determinism across launch geometry - the axis the existing same-batch-twice
# compare cannot see. It launches the shipped kernel directly, so it needs the
# internal header from src/ (like bench/ does).
cudec_add_test(determinism_gpu SOURCES determinism_gpu.cu LIBS
               cudec_test_fixtures GPU)
if(TARGET determinism_gpu)
  target_include_directories(determinism_gpu
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()

# ---- Structural conformance, configure time: a violation reds every CI
# build. These are drift detectors, not tamper-proofing - the honest limits
# are recorded in docs/MASTERPLAN.md section 5.

# The library must depend on nothing beyond the CUDA runtime - checked on
# the private AND consumer-facing link surfaces (an INTERFACE dependency of
# a static library is exactly what its consumers inherit), and on link
# options, which could smuggle a -l dependency past the target lists.
foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES LINK_OPTIONS
              INTERFACE_LINK_OPTIONS)
  get_target_property(_entries cudec ${_prop})
  if(_entries)
    foreach(_entry IN LISTS _entries)
      # A PRIVATE link of an imported target lands in the interface list
      # wrapped as $<LINK_ONLY:CUDA::cudart>; strip that wrapper before
      # matching so CUDA::cudart (the CUDA runtime the rule permits) is not
      # a false positive.
      string(REGEX REPLACE "^\\$<LINK_ONLY:(.+)>$" "\\1" _bare "${_entry}")
      if(NOT _bare MATCHES "^CUDA::")
        message(
          FATAL_ERROR
            "cudec carries '${_entry}' in ${_prop} - the library must depend "
            "on nothing beyond the CUDA runtime")
      endif()
    endforeach()
  endif()
endforeach()

# Warnings-as-errors must be wired on the library...
get_target_property(_cudec_opts cudec COMPILE_OPTIONS)
if(NOT _cudec_opts MATCHES "Werror")
  message(FATAL_ERROR "warnings-as-errors is no longer wired on target cudec")
endif()

# ...and LIVE: a translation unit with a known warning must FAIL to compile
# under the exact flag list the targets use, invoked directly (try_compile
# does not forward per-language flags reliably - verified empirically). The
# presence check above cannot see a dead gate (an appended -Wno-error still
# contains the substring); these probes can. Honest scope limit: nvcc's
# front end diagnoses both probes (#177-D), so this proves the
# --Werror=all-warnings seam end to end; the -Xcompiler=-Werror suffix has
# no cheap gcc-only trigger (the front end shadows every candidate tried)
# and stays covered by the substring check only.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/host_warning.cu
     "int main() {\n    int unused_host_variable;\n    return 0;\n}\n")
file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/device_warning.cu
  "__global__ void probe() { int unused_device_variable; }\nint main() { return 0; }\n"
)
# Positive control first: a warning-free TU must compile with exit 0 under
# the identical command. Without it, ANY probe breakage (arch renamed in a
# toolkit bump, unwritable dir, rejected flag) exits nonzero and would
# masquerade as "gate live" - the probes would go vacuous exactly on the
# toolchain-drift event they guard against.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.cu
     "int main() { return 0; }\n")
execute_process(
  COMMAND
    ${CMAKE_CUDA_COMPILER} ${CUDEC_CUDA_STRICT_FLAGS_RAW} -arch=sm_86 -c
    ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.cu -o
    ${CMAKE_CURRENT_BINARY_DIR}/probes/control_clean.o
  RESULT_VARIABLE _control_exit
  OUTPUT_QUIET ERROR_QUIET)
if(NOT _control_exit EQUAL 0)
  message(
    FATAL_ERROR
      "warnings-gate probe invocation is broken (the warning-free control "
      "TU failed to compile) - fix the probe before trusting the gate")
endif()
foreach(_probe host_warning device_warning)
  execute_process(
    COMMAND
      ${CMAKE_CUDA_COMPILER} ${CUDEC_CUDA_STRICT_FLAGS_RAW} -arch=sm_86 -c
      ${CMAKE_CURRENT_BINARY_DIR}/probes/${_probe}.cu -o
      ${CMAKE_CURRENT_BINARY_DIR}/probes/${_probe}.o
    RESULT_VARIABLE _probe_exit
    OUTPUT_QUIET ERROR_QUIET)
  if(_probe_exit EQUAL 0)
    message(
      FATAL_ERROR
        "warnings-as-errors gate is dead: probe '${_probe}' compiled clean "
        "under the strict flags")
  endif()
endforeach()

# Single-source lock (issue #40): the host-side CUDA RAII owners live once, in
# src/cuda_raii.h, and every CUDA host translation unit reuses them instead of
# carrying its own copy. This detector reds the build if the header stops
# defining an owner, or if a host TU drops the include or re-introduces a local
# owner definition. Honest limit (as with the checks above): it catches a
# re-copy under the SAME struct names, not one smuggled in under a fresh name -
# a drift detector, not tamper-proofing.
set(_raii_header ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_raii.h)
if(NOT EXISTS ${_raii_header})
  message(FATAL_ERROR "src/cuda_raii.h is missing: the shared CUDA RAII owners "
                      "must be single-sourced there (issue #40)")
endif()
file(READ ${_raii_header} _raii_src)
foreach(_owner DevBuf PinnedBuf StreamOwner EventOwner)
  if(NOT _raii_src MATCHES "struct ${_owner}")
    message(FATAL_ERROR "src/cuda_raii.h no longer defines the shared owner "
                        "'${_owner}' (issue #40)")
  endif()
endforeach()
foreach(_tu frame.cpp stream.cpp)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../src/${_tu} _tu_src)
  if(NOT _tu_src MATCHES "#include \"cuda_raii.h\"")
    message(FATAL_ERROR "src/${_tu} no longer includes cuda_raii.h: it must "
                        "reuse the shared CUDA RAII owners (issue #40)")
  endif()
  foreach(_owner DevBuf PinnedBuf StreamOwner EventOwner DevPtr)
    if(_tu_src MATCHES "struct ${_owner}")
      message(FATAL_ERROR "src/${_tu} defines a local 'struct ${_owner}': the "
                          "CUDA RAII owners are single-sourced in cuda_raii.h "
                          "(issue #40)")
    endif()
  endforeach()
endforeach()

# ---- Termination, warp-collective integrity, and determinism as structural
# rules (issue #72). Same honest scope as the checks above: drift detectors
# under the known spellings, not tamper-proofing. Every rule below is applied
# by ONE scanning function, which is then run over inline probe files that
# must and must not trip it - a detector that silently stopped matching would
# reassure forever, exactly like a dead warnings gate.

# Scans one source and returns every structural violation it finds. Comments
# are stripped first, so prose ("never double-frees", "decode loops
# terminate") can neither trip a rule nor mask one. `check_loops` selects the
# termination rule, which is scoped to the decode path rather than to every
# helper.
#
# Rules:
#  - loops: a counted `for (init; cond; incr)` is accepted on its visible
#    induction variable; every other loop form (`while`, `do {`, a `for`
#    header with an empty condition) must name an explicit decrementing
#    `fuel` counter on its own line, the line above it, or the three below,
#    because a text scan cannot verify a monotone-cursor argument and a
#    non-terminating decode hangs the device (nvCOMP 5.3's release notes
#    record shipping exactly that on malformed Snappy input);
#  - warp collectives: the legacy non-_sync intrinsics are banned outright
#    (since Volta's independent thread scheduling they cannot express which
#    lanes participate at all), and a collective's mask must be a full-warp
#    constant or __activemask(), never a value named after a parsed bitstream
#    field - corrupting collective participation metadata is a published
#    attack on CUDA kernels that leaves control flow untouched;
#  - determinism: no floating-point type and no floating-point atomic, which
#    is what makes the gpu_to_gpu level in docs/DETERMINISM.md hold by
#    construction rather than by measurement.
#
# Limits, stated as the neighbouring checks state theirs: constructs hidden
# inside string literals are not stripped; a mask computed into a variable on
# an earlier line is judged by its name at the call site; and a loop or
# intrinsic introduced under a fresh spelling is code review's job.
function(cudec_scan_source path check_loops out_violations)
  set(_violations "")
  set(_in_comment FALSE)
  set(_pending_reason "")
  set(_pending_line 0)
  set(_pending_left 0)
  set(_previous "")
  set(_lineno 0)
  file(STRINGS "${path}" _lines)
  foreach(_line IN LISTS _lines)
    math(EXPR _lineno "${_lineno} + 1")
    if(_in_comment)
      if(_line MATCHES "\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _line "${_line}")
        set(_in_comment FALSE)
      else()
        set(_line "")
      endif()
    endif()
    string(REGEX REPLACE "//.*$" " " _line "${_line}")
    if(_line MATCHES "/\\*")
      string(REGEX REPLACE "/\\*.*$" " " _head "${_line}")
      if(_line MATCHES "/\\*.*\\*/")
        string(REGEX REPLACE "^.*\\*/" " " _tail "${_line}")
        set(_line "${_head} ${_tail}")
      else()
        set(_line "${_head}")
        set(_in_comment TRUE)
      endif()
    endif()
    set(_code " ${_line} ")

    if(check_loops)
      if(_pending_left GREATER 0)
        if(_code MATCHES "fuel")
          set(_pending_left 0)
        else()
          math(EXPR _pending_left "${_pending_left} - 1")
          if(_pending_left EQUAL 0)
            string(APPEND _violations
                   "  ${path}:${_pending_line}: ${_pending_reason} with no "
                   "explicit decrementing 'fuel' cap\n")
          endif()
        endif()
      endif()
      set(_form "")
      if(_code MATCHES "\\}[ \t]*while")
        # The tail of a do-while; its `do {` carries the requirement.
      elseif(_code MATCHES "[^A-Za-z_0-9]while[ \t]*\\(")
        set(_form "a 'while' loop")
      elseif(_code MATCHES "[^A-Za-z_0-9]do[ \t]*{")
        set(_form "a 'do' loop")
      elseif(_code MATCHES "[^A-Za-z_0-9]for[ \t]*\\([^)]*;[ \t]*;")
        set(_form "a 'for' loop with an empty condition")
      elseif(_code MATCHES "[^A-Za-z_0-9]for[ \t]*\\([ \t]*;")
        set(_form "a 'for (;;)' loop")
      elseif(_code MATCHES "[^A-Za-z_0-9]for[ \t]*\\(")
        string(REPLACE ";" "" _no_semis "${_code}")
        string(LENGTH "${_code}" _with_len)
        string(LENGTH "${_no_semis}" _without_len)
        math(EXPR _semis "${_with_len} - ${_without_len}")
        if(_semis LESS 2)
          set(_form "a 'for' header without a visible condition and increment")
        endif()
      endif()
      if(_form AND NOT _code MATCHES "fuel" AND NOT _previous MATCHES "fuel")
        set(_pending_reason "${_form}")
        set(_pending_line ${_lineno})
        set(_pending_left 3)
      endif()
      set(_previous "${_code}")
    endif()

    foreach(_legacy __shfl __shfl_up __shfl_down __shfl_xor __ballot __any
                    __all __match_any)
      if(_code MATCHES "${_legacy}[ \t]*\\(")
        string(APPEND _violations
               "  ${path}:${_lineno}: legacy warp intrinsic '${_legacy}(' - "
               "only the _sync form can state which lanes participate\n")
      endif()
    endforeach()

    if(_code MATCHES "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(")
      string(REGEX MATCH "(__[A-Za-z_0-9]*_sync|__syncwarp)[ \t]*\\(([^,)]*)"
                   _call "${_code}")
      set(_collective "${CMAKE_MATCH_1}")
      string(STRIP "${CMAKE_MATCH_2}" _mask)
      if(NOT _mask STREQUAL "" AND NOT _mask MATCHES
         "^(0[xX][fF][fF][fF][fF][fF][fF][fF][fF][uU]?|kFullWarpMask|__activemask\\(?)$")
        string(APPEND _violations
               "  ${path}:${_lineno}: '${_collective}' mask '${_mask}' is "
               "neither the full-warp constant nor __activemask()\n")
      endif()
      foreach(_field token offset literals_len match_len src_pos dst_pos)
        if(_code MATCHES "[^A-Za-z_0-9]${_field}[^A-Za-z_0-9]")
          string(APPEND _violations
                 "  ${path}:${_lineno}: '${_collective}' names the parsed "
                 "bitstream field '${_field}' - participation metadata must "
                 "never come from unvalidated input\n")
        endif()
      endforeach()
      if(_code MATCHES "[^A-Za-z_0-9](seq|sequence|parser)\\.")
        string(APPEND _violations
               "  ${path}:${_lineno}: '${_collective}' reads a parsed "
               "sequence field - participation metadata must never come from "
               "unvalidated input\n")
      endif()
    endif()

    foreach(_fp float double __half __half2 __nv_bfloat16 __nv_bfloat162)
      if(_code MATCHES "[^A-Za-z_0-9]${_fp}[^A-Za-z_0-9]")
        string(APPEND _violations
               "  ${path}:${_lineno}: floating-point type '${_fp}' - the "
               "decode path is integer-only, which is what makes the "
               "gpu_to_gpu determinism level hold (docs/DETERMINISM.md)\n")
      endif()
    endforeach()
    if(_code MATCHES "atomic[A-Za-z]*[ \t]*\\(" AND _code MATCHES
                                                   "(float|double|__half|__nv_bfloat16)")
      string(APPEND _violations
             "  ${path}:${_lineno}: floating-point atomic - its result "
             "depends on the order the device happens to schedule\n")
    endif()
  endforeach()
  if(check_loops AND _pending_left GREATER 0)
    string(APPEND _violations
           "  ${path}:${_pending_line}: ${_pending_reason} with no explicit "
           "decrementing 'fuel' cap\n")
  endif()
  set(${out_violations} "${_violations}" PARENT_SCOPE)
endfunction()

# Liveness probes for the scanner itself, in the same spirit as the
# warnings-gate control TU above: the clean probe must come back silent
# (including its comment full of the very keywords the rules match), and each
# violating probe must come back naming its rule. These files are scanned as
# text and never compiled.
set(_probe_dir ${CMAKE_CURRENT_BINARY_DIR}/probes)
file(
  WRITE ${_probe_dir}/scan_clean.cu
  "/* Prose naming while, do, for, float, double, __shfl( and a mask from a\n"
  " * token - the comment stripper must remove all of it. */\n"
  "unsigned probe_counted(unsigned n) {\n"
  "    unsigned total = 0;\n"
  "    for (unsigned i = 0; i < n; i++) {\n"
  "        total += i;\n"
  "    }\n"
  "    unsigned fuel = n + 1;\n"
  "    while (fuel-- != 0) {\n"
  "        total += 1;\n"
  "    }\n"
  "    unsigned left = n;\n"
  "    unsigned fuel_drain = n + 1;\n"
  "    do {\n"
  "        if (fuel_drain-- == 0) {\n"
  "            break;\n"
  "        }\n"
  "        left = left - 1;\n"
  "    } while (left != 0);\n"
  "    return total;\n"
  "}\n"
  "__device__ int probe_collective(int v) {\n"
  "    __syncwarp();\n"
  "    return __shfl_sync(0xFFFFFFFFu, v, 0);\n"
  "}\n")
file(
  WRITE ${_probe_dir}/scan_unbounded_loop.cu
  "unsigned probe(const unsigned char* src, unsigned n) {\n"
  "    unsigned count = src[0];\n"
  "    while (count != 0) {\n"
  "        count += src[count % n];\n"
  "    }\n"
  "    return count;\n"
  "}\n")
file(WRITE ${_probe_dir}/scan_legacy_intrinsic.cu
     "__device__ int probe(int v) {\n" "    return __shfl(v, 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_derived_mask.cu
     "__device__ int probe(int v, unsigned mask_bits) {\n"
     "    return __shfl_sync(mask_bits, v, 0);\n" "}\n")
file(WRITE ${_probe_dir}/scan_floating_point.cu
     "__device__ void probe(float* acc, float v) {\n"
     "    atomicAdd(acc, v);\n" "}\n")

cudec_scan_source(${_probe_dir}/scan_clean.cu TRUE _probe_clean)
if(_probe_clean)
  message(
    FATAL_ERROR
      "the structural scanner rejects a compliant probe - fix the scanner "
      "before trusting it:\n${_probe_clean}")
endif()
foreach(_probe scan_unbounded_loop scan_legacy_intrinsic scan_derived_mask
               scan_floating_point)
  cudec_scan_source(${_probe_dir}/${_probe}.cu TRUE _probe_result)
  if(NOT _probe_result)
    message(FATAL_ERROR "the structural scanner is dead: probe '${_probe}' "
                        "passed clean (issue #72)")
  endif()
endforeach()

# The decode path: the sources whose loops are driven by lengths and counts
# read out of a hostile bitstream. Every device translation unit must be on
# this list, so a new kernel cannot quietly escape the termination rule; the
# checksum helper (src/xxhash32.h) is deliberately not on it - its loops walk
# a caller-supplied buffer at a fixed stride with no stream-derived exit.
set(_decode_path_sources lz4_block.h lz4_decode.cuh frame.cpp batch.cu
                         stream.cpp)
file(GLOB _device_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../src
     CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cuh)
foreach(_device_source IN LISTS _device_sources)
  if(NOT _device_source IN_LIST _decode_path_sources)
    message(
      FATAL_ERROR
        "src/${_device_source} is device code but is not on the decode-path "
        "scan list in tests/CMakeLists.txt - add it (issue #72)")
  endif()
endforeach()

set(_structural_violations "")
foreach(_source IN LISTS _decode_path_sources)
  set(_path ${CMAKE_CURRENT_SOURCE_DIR}/../src/${_source})
  if(NOT EXISTS ${_path})
    message(FATAL_ERROR "src/${_source} is on the decode-path scan list but "
                        "does not exist (issue #72)")
  endif()
  cudec_scan_source(${_path} TRUE _found)
  string(APPEND _structural_violations "${_found}")
endforeach()
# The remaining sources are held to the collective and floating-point rules
# only; the termination rule is scoped above.
file(GLOB _other_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../src
     CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.c
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.h)
foreach(_source IN LISTS _other_sources)
  if(NOT _source IN_LIST _decode_path_sources)
    cudec_scan_source(${CMAKE_CURRENT_SOURCE_DIR}/../src/${_source} FALSE
                      _found)
    string(APPEND _structural_violations "${_found}")
  endif()
endforeach()
if(_structural_violations)
  message(FATAL_ERROR "structural rules violated (issue #72):\n"
                      "${_structural_violations}")
endif()

# Finally: no ctest entry in this directory may be without a timeout.
cudec_assert_test_timeouts()
