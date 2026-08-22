#!/bin/sh
# Divides ONE wall-clock fuzzing budget across the declared targets in
# proportion to a share declared beside each target, and prints the table it
# derived.
#
# The failure this prevents is the one the flat per-target budget walked into.
# A budget written as "seconds per target" is multiplied by a target count
# nobody re-reads, so the job's wall clock grows every time a target lands. The
# fuzz job's own timeout does not grow with it, and the run that crosses it is
# killed with no findings reported and no seed corpus carried out - a red cron
# whose cause is in the runner's clock rather than in any parser. Measured
# rather than supposed: the last scheduled run declared 900 s per target with
# two targets declared, and took 30 m 31 s under a 50-minute job timeout:
#
#   gh run list --repo iderex/cudec --workflow fuzz.yml --event schedule \
#     --limit 10 --json databaseId,conclusion,createdAt,updatedAt
#   31927499342  success  2026-08-16T04:48:22Z -> 2026-08-16T05:18:53Z
#
#   git show f38b542d007cc488b05b4937fbc20ddaf9c0c894:fuzz/CMakeLists.txt |
#     sed -n 's/^cudec_add_fuzz_target(\([A-Za-z0-9_]*\).*/\1/p' |
#     grep -v '_selftest$' | LC_ALL=C sort -u | wc -l
#   2
#
# Five targets have landed since. With a total instead of a per-target number,
# the next target changes how the budget is SPLIT and never how long the job
# runs.
#
# THE SHARES ARE NOT EQUAL, AND THAT IS THE SECOND REASON THIS EXISTS. A target
# over an envelope raw bytes reach directly saturates early: the grammar above
# it is a few header fields and the engine finds them in seconds. A
# structure-aware target spends its budget inside a table or a whole frame
# decode, where the reachable state grows with the time it is given. Splitting
# a fixed budget evenly across the two spends the same seconds where they stop
# buying anything and starves the place they still do.
#
# The shares are declared at the target, in fuzz/CMakeLists.txt, and never
# here: this script must not hold a list of targets, because a list here would
# drift against the build that decides which targets exist, and the drift is
# silent in the dangerous direction - a target the list forgot would get no
# time while the job reads green.
#
# Three refusals:
#
#   - a declared target with no SHARE, or a SHARE that is not a positive
#     integer, so a target cannot land without saying what it is worth;
#   - a total that is not a positive integer;
#   - a derived per-target budget of zero seconds, which libFuzzer accepts and
#     runs as an immediate exit, so a target squeezed out of the budget would
#     read exactly like a target that was fuzzed and found nothing.
#
# The deriver proves itself on every run before it is trusted with the real
# file, the way scripts/check-fuzz-corpus.sh does: it runs first over a copy of
# fuzz/CMakeLists.txt with one target's SHARE deleted, and a run that ACCEPTS
# that copy is a run whose table means nothing.
set -eu

total="${1:-}"
cmakelists="${2:-fuzz/CMakeLists.txt}"

die() {
    echo "fuzz-budget: $1" >&2
    exit 1
}

case "$total" in
'' | *[!0-9]*)
    die "the first argument is the total budget in whole seconds, and '$total'
is not one. This script refuses rather than choosing a total of its own."
    ;;
esac
[ "$total" -gt 0 ] || die "a total budget of $total seconds fuzzes nothing"

[ -f "$cmakelists" ] ||
    die "$cmakelists does not exist, so the target list cannot be derived.
This check refuses rather than falling back to a list of its own."

# The deriver, over a file that is a fuzz/CMakeLists.txt. Called twice: once on
# a planted defect, once on the real thing. Names are taken from the start of
# the line so the function's own definition and the prose above it cannot
# contribute one; the _selftest twins are skipped because they are replayed
# against the committed seeds rather than fuzzed, and own no budget.
derive() {
    awk -v total="$total" '
        /^cudec_add_fuzz_target\(/ {
            line = $0
            sub(/^cudec_add_fuzz_target\(/, "", line)
            n = split(line, field, /[ \t()]+/)
            name = field[1]
            if (name ~ /_selftest$/) next
            if (name in seen) next
            seen[name] = 1

            share = ""
            for (i = 2; i <= n; i++) {
                if (field[i] == "SHARE") { share = field[i + 1]; break }
            }
            if (share !~ /^[0-9]+$/ || share + 0 == 0) {
                msg = "target " name " declares no positive SHARE"
                print msg > "/dev/stderr"
                bad = 1
                next
            }
            order[++count] = name
            weight[name] = share + 0
            sum += share + 0
        }
        END {
            if (bad) exit 1
            if (count == 0) {
                msg = "no cudec_add_fuzz_target(...) call found"
                print msg > "/dev/stderr"
                exit 1
            }
            for (i = 1; i <= count; i++) {
                name = order[i]
                seconds = int(total * weight[name] / sum)
                if (seconds == 0) {
                    printf "%s would be fuzzed for 0 s out of a %d s total\n", \
                        name, total > "/dev/stderr"
                    exit 1
                }
                printf "%s %d\n", name, seconds
            }
        }
    ' "$1"
}

# The self-test. One target loses its SHARE in a copy, which is exactly the
# shape this refuses; a deriver that accepts it would hand every later step a
# table with a target missing from it and no sign that one is missing.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# Which target to plant the defect in is taken from the file directly rather
# than from a derivation this script has not yet proved: a self-test whose
# subject came out of the thing under test cannot report the real cause when
# the thing under test is what is broken.
first_target="$(
    sed -n 's/^cudec_add_fuzz_target(\([A-Za-z0-9_]*\).*/\1/p' "$cmakelists" |
        grep -v '_selftest$' |
        head -n 1
)"
[ -n "$first_target" ] ||
    die "no cudec_add_fuzz_target(...) call was found in $cmakelists.
An empty target list would leave the whole budget unspent and the job green."
sed "/^cudec_add_fuzz_target($first_target /s/ SHARE [0-9]*//" \
    "$cmakelists" >"$work/planted"
cmp -s "$cmakelists" "$work/planted" &&
    die "the planted copy is identical to $cmakelists, so the self-test would
prove nothing. The SHARE of '$first_target' was not where this expected it."
if derive "$work/planted" >"$work/self-test.log" 2>&1; then
    cat "$work/self-test.log" >&2
    die "the self-test file was accepted. Its '$first_target' target declares
no SHARE at all, so a deriver that accepts it is refusing nothing and the table
it prints for the real file means nothing."
fi
echo "PASS: self-test - the deriver refuses a target that declares no SHARE" >&2

derive "$cmakelists" || die "$cmakelists does not declare a usable budget"
