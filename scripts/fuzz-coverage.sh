#!/bin/sh
# Runs one fuzz target for a fixed wall-clock budget from the committed seeds
# and prints the edge coverage it reached, as one line a record can hold.
#
# WHY THIS EXISTS AT ALL, since the fuzz job already runs every target. Issue
# #193 is decided by a coverage comparison - a structure-aware mutation layer
# is KEPT or DROPPED on whether it reaches materially more of the table and
# page-decode targets than the plain byte fuzzer does in the same budget - and
# nothing in this tree emitted a coverage number a comparison could be made
# from. The numbers were there and unreachable: libFuzzer prints `cov:` and
# `ft:` on its own progress lines, which land in a job log that is rotated and
# is not a record. This script is the route from that line to a figure with a
# command behind it.
#
# NO NEW INSTRUMENTATION, DELIBERATELY. A `-fprofile-instr-generate` build plus
# llvm-profdata and llvm-cov would give region and line coverage, and would be a
# SECOND instrumented build of the same sources measuring a different thing from
# what the engine actually steers on. `cov:` is the count of coverage-instrumented
# edges the run executed, from the same instrumentation the mutator's lift would
# show up in, so it is both the cheaper and the more faithful number here. What
# it cannot do is attribute an edge to a source line, and nothing below claims
# it can.
#
# WHAT A NUMBER FROM THIS IS AND IS NOT. It is one run of a stochastic search,
# so a single pair of numbers is not a comparison: the caller runs several
# libFuzzer seeds per arm and reads the spread. The script takes the seed as an
# argument rather than choosing one, so a run is reproducible and two arms can
# be given the same one.
#
#   scripts/fuzz-coverage.sh <build-dir> <target> <seconds> <seed>
#
# Four refusals, each for a way a coverage run reports a number about nothing:
#
#   - a missing binary, so a stale build directory cannot answer for a target
#     it never built;
#   - a corpus directory that is absent or empty, which libFuzzer reads as a
#     legal start-from-scratch run - the arms would then not share their seeds
#     and the comparison would be against a different experiment;
#   - a run whose output carries no DONE line, which is what a crash, an
#     out-of-memory kill or a signal leaves behind: the run stopped early and
#     its last coverage figure is not the budget's;
#   - a parsed coverage of zero, which no instrumented binary reaches after
#     executing a single input, so it means the parse missed rather than that
#     the target covered nothing.
#
# The parser proves it refuses before it is trusted, the way
# scripts/fuzz-budget.sh and scripts/check-fuzz-corpus.sh do: it is run first
# over a planted log with the DONE line removed, and a parser that accepts that
# is a parser whose figure means nothing.
set -eu

builddir="${1:-}"
target="${2:-}"
seconds="${3:-}"
seed="${4:-}"

die() {
    echo "fuzz-coverage: $1" >&2
    exit 1
}

[ -n "$builddir" ] && [ -n "$target" ] ||
    die "usage: fuzz-coverage.sh <build-dir> <target> <seconds> <seed>"

case "$seconds" in
'' | *[!0-9]*) die "the third argument is a whole number of seconds, not '$seconds'" ;;
esac
[ "$seconds" -gt 0 ] || die "a budget of $seconds seconds measures nothing"

case "$seed" in
'' | *[!0-9]*) die "the fourth argument is libFuzzer's -seed, not '$seed'" ;;
esac

bin="$builddir/fuzz/$target"
[ -x "$bin" ] || die "$bin is not an executable, so this build directory
cannot answer for $target. Refusing rather than reporting a number from
whatever else is in it."

corpus="fuzz/corpus/$target"
[ -d "$corpus" ] || die "$corpus does not exist"
[ "$(find "$corpus" -type f | wc -l)" -gt 0 ] ||
    die "$corpus holds no seed. libFuzzer reads that as a legal
start-from-scratch run, so the arms of a comparison would not share their
seeds."

# The one line libFuzzer prints when it stops on its own clock, and the only
# line a figure is taken from. `-runs` and `-max_total_time` both end in DONE;
# a crash, an OOM or a signal does not.
parse() {
    awk '
        /DONE/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "cov:") cov = $(i + 1)
                if ($i == "ft:") ft = $(i + 1)
            }
            execs = $1
            sub(/^#/, "", execs)
            seen = 1
        }
        END {
            if (!seen) { print "no DONE line" > "/dev/stderr"; exit 1 }
            if (cov + 0 == 0) { print "parsed cov: 0" > "/dev/stderr"; exit 1 }
            printf "%s %s %s\n", cov, ft, execs
        }
    ' "$1"
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The self-test. A log that stopped before DONE is exactly what a killed run
# leaves, and a parser that answered from its last progress line instead would
# report a partial run's coverage as the budget's.
planted="$work/planted.log"
printf '#9\tINITED cov: 1 ft: 1 corp: 1/1b\n#99\tNEW    cov: 2 ft: 2\n' > "$planted"
if parse "$planted" >/dev/null 2>&1; then
    die "the parser accepted a log with no DONE line, so its figures would
include runs that stopped early. Refusing before measuring anything."
fi

log="$work/$target.log"
# The seeds are given read-only behind a scratch directory, the way the fuzz
# workflow drives them, so a measurement cannot rewrite the corpus its own
# comparison rests on.
mkdir -p "$work/corpus"
"$bin" "$work/corpus" "$corpus" \
    -max_total_time="$seconds" \
    -max_len=4096 \
    -seed="$seed" \
    -print_final_stats=1 > "$log" 2>&1 || {
    echo "fuzz-coverage: $target exited non-zero; last 20 lines:" >&2
    tail -n 20 "$log" >&2
    exit 1
}

figures="$(parse "$log")" || die "$target produced no readable DONE line"
set -- $figures
printf '%s seed=%s seconds=%s cov=%s ft=%s execs=%s\n' \
    "$target" "$seed" "$seconds" "$1" "$2" "$3"
