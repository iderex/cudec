#!/bin/sh
# Runs all four Compute Sanitizer tools over every gpu-labeled ctest target
# and prints one paste-ready PASS/FAIL block naming the machine it ran on.
#
# It is fail-closed on a VACUOUS run: a sanitizer sweep that checked nothing
# reads exactly like a clean one. An absent compute-sanitizer, an empty
# discovery, a missing test binary, and a known GPU test gone from the
# discovered set are hard errors here, never skips.
#
# Discovery is `ctest -L gpu -N`, never a list in this file, so a GPU test is
# covered the moment it is registered with the label.
#
# Usage, from the repository root, inside the pinned container (the image
# ships no cmake and no git, hence the apt-get and CUDEC_COMMIT):
#
#   docker run --rm --gpus all -v "$PWD:/w" -w /w \
#     nvidia/cuda:12.6.2-devel-ubuntu24.04@sha256:738fba0fbdb225b7a2931c58a5c8f03a84d3cd2f6a84975826a157339ef750b8 \
#     sh -c "apt-get update -q >/dev/null && \
#            apt-get install -yq cmake >/dev/null 2>&1 && \
#            cmake -B build-cuda -DCUDEC_ENABLE_CUDA=ON && \
#            cmake --build build-cuda -j && \
#            CUDEC_COMMIT=<sha> scripts/sanitize-gpu.sh build-cuda"
#
# The tool flags are pinned against `compute-sanitizer --help` of the version
# in that image, 2024.3.0.0 (build 34841621). Two flags named while this
# script was specified do not exist there, and the substitutes are stated
# rather than assumed:
#
#   - memcheck has no --report-device-side-allocation-failure. Device heap
#     allocation is covered by --check-device-heap, passed explicitly below
#     although it already defaults to yes, so the run states the option
#     instead of implying it.
#   - initcheck has no --initcheck-address-space. Its own help calls it
#     "Global memory initialization checking", so in this version the tool
#     has no shared-memory address space to switch on; shared memory is
#     racecheck's subject ("Shared memory hazard checking") and racecheck
#     runs below. initcheck's remaining knob, --track-unused-memory, reports
#     unused allocations rather than uninitialised reads and is left off.
#
# Re-confirm both against --help on any toolkit bump. A flag this script
# passes that the tool does not know is a hard error from compute-sanitizer
# itself rather than a silent skip.
set -eu

build="${1:-build-cuda}"

die() {
    echo "sanitize-gpu: $1" >&2
    exit 1
}

command -v compute-sanitizer >/dev/null 2>&1 ||
    die "compute-sanitizer is not on PATH - this must run inside the pinned
CUDA container (see the header). A sweep that ran no tool is refused."
command -v ctest >/dev/null 2>&1 ||
    die "ctest is not on PATH - discovery is impossible, so the run would
cover nothing."
command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi is not on PATH - the summary must name the GPU and driver
it ran against, so an unattributable run is refused."
command -v nvcc >/dev/null 2>&1 ||
    die "nvcc is not on PATH - the summary must name the CUDA version it ran
against, so an unattributable run is refused."

[ -d "$build" ] || die "build directory '$build' does not exist"

# The commit is part of the evidence: a pasted block naming no commit cannot
# be checked against anything. The container ships no git, and a git
# worktree's .git is a file naming a host path that does not resolve inside
# the container, so the sha is passed in and derived from git only as a
# convenience for a host-side run.
commit="${CUDEC_COMMIT:-}"
if [ -z "$commit" ]; then
    commit="$(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || true)"
fi
[ -n "$commit" ] ||
    die "no tested commit: set CUDEC_COMMIT=\$(git rev-parse HEAD) on the
host, where git can read the repository."

# Discovery, in the shape .github/workflows/ci.yml lists the deferred GPU
# tests with. ctest prints "  Test #3: smoke"; anything else on a line is not
# a test name.
listing="$(ctest --test-dir "$build" -L gpu -N)" ||
    die "ctest discovery failed in '$build'"
tests="$(printf '%s\n' "$listing" |
    sed -n 's/^ *Test *#[0-9][0-9]*: *\([^ ][^ ]*\) *$/\1/p')"
[ -n "$tests" ] ||
    die "zero gpu-labeled tests discovered in '$build' - a sweep over nothing
is not a clean sweep. Configure with -DCUDEC_ENABLE_CUDA=ON."

# The cross-check .github/workflows/ci.yml already makes on the same
# listing, by name: a known GPU test quietly leaving the discovered set (a
# dropped label, a renamed target, a registration lost in a merge) would
# otherwise shrink this sweep and still print PASS. The names are that job's,
# so the two gates cannot drift apart unnoticed.
for known in smoke gpu_fixture frame_twin stream_twin termination_gpu \
    determinism_gpu; do
    printf '%s\n' "$tests" | grep -qx "$known" ||
        die "gpu test '$known' is absent from the discovered set - the sweep
would cover less than the CI job's own cross-check demands."
done

# Name to binary, searched rather than assumed at tests/<name>: a moved
# output directory becomes a hard error here instead of a wrong path later,
# and two matches are as unusable as none.
binary_of() {
    matches="$(find "$build" -type f -perm -u+x -name "$1" 2>/dev/null ||
        true)"
    count="$(printf '%s' "$matches" | grep -c . || true)"
    [ "$count" = "1" ] ||
        die "expected exactly one executable named '$1' under '$build', found
$count - discovery and the build tree disagree."
    printf '%s\n' "$matches"
}

# The tool's own flags; --error-exitcode 1 is appended to every invocation,
# because a sanitizer that reports an error and still exits 0 is the vacuous
# run in another shape.
flags_of() {
    case "$1" in
    memcheck) echo "--tool memcheck --leak-check full --check-device-heap yes" ;;
    racecheck) echo "--tool racecheck" ;;
    initcheck) echo "--tool initcheck --check-api-memory-access yes" ;;
    synccheck) echo "--tool synccheck" ;;
    *) die "unknown tool '$1'" ;;
    esac
}

# The summary is accumulated in a file rather than a variable: every line is
# written the moment its tool exits, so an interrupted run still leaves the
# results of what did run.
results="$(mktemp)"
trap 'rm -f "$results"' EXIT

report() {
    echo
    echo "=== Compute Sanitizer summary ==="
    echo "commit:    $commit"
    echo "gpu:       $(nvidia-smi --query-gpu=name,driver_version \
        --format=csv,noheader)"
    echo "cuda:      $(nvcc --version | sed -n \
        's/^.*release \([0-9][0-9.]*\).*$/\1/p')"
    echo "sanitizer: $(compute-sanitizer --version |
        sed -n 's/^Version //p')"
    echo "build:     $build"
    echo
    cat "$results"
    echo
}

failed=""
for name in $tests; do
    bin="$(binary_of "$name")" || exit 1
    for tool in memcheck racecheck initcheck synccheck; do
        flags="$(flags_of "$tool")"
        echo "=== $name : $tool ==="
        # Unquoted on purpose: $flags is this file's own constant list.
        if compute-sanitizer $flags --error-exitcode 1 "$bin"; then
            printf '  %-16s %-10s PASS\n' "$name" "$tool" >>"$results"
        else
            printf '  %-16s %-10s FAIL\n' "$name" "$tool" >>"$results"
            failed="$name/$tool"
            break
        fi
    done
    [ -z "$failed" ] || break
done

report

if [ -n "$failed" ]; then
    echo "sanitize-gpu: FAILED at $failed" >&2
    exit 1
fi
echo "sanitize-gpu: clean over $(printf '%s\n' "$tests" | grep -c .) gpu \
tests, four tools each"
