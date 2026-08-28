#!/bin/sh
# Refuses a tense-present tracked document that names a repository path the
# tree does not hold.
#
# The failure this prevents is a document asserting something about the tree
# that the tree refutes. Three such corrections landed by hand in one night -
# #391 (a decode kernel named by a source file that had been renamed), #393
# and #395 - and nothing in this repository would have found any of them.
#
# WHAT THIS REACHES IS ONE THIRD OF THAT CLASS, AND THE OTHER TWO THIRDS ARE
# WHY A GREEN RUN HERE IS NOT "NO DOCUMENT CONTRADICTS THE TREE". #391 is a
# path reference and this refuses it. #393 - a README row reading `planned`
# beside a milestone whose issues are all closed - and #395 - a sentence
# saying a rung has not landed while docs/BENCHMARKS.md carries its numbers -
# are claims about STATE rather than references to files, and no reading of a
# path reaches either. Whether those are refusable at all is a separate
# question this check does not ask.
#
# Two properties, per the scope call taken on #397:
#
#   1. A path a tense-present document names must resolve.
#   2. A declared forward reference - `path/to/file` (planned: #NNN) - must
#      name an issue that is still open.
#
# The second exists because a document may legitimately name a file the tree
# OWES: src/gdeflate_tables.h and src/gdeflate_block.h are named as their
# surface by #175, #176 and #182, and an M4 design paragraph naming them would
# be honest text that property 1 reds. The outright-refusal reading - a design
# section names the issue instead of the file - was refused because a
# masterplan that may not name the shape of the tree it is planning is a
# masterplan fighting its own check.
#
# The declaration is not a grace period. The moment the issue it names closes
# without the path existing, the reference reds exactly as a broken
# present-tense path would, so a promise cannot outlive the work it pointed
# at. Undeclared future tense - a path that does not resolve and carries no
# declaration - is refused outright.
#
# THREE CARVE-OUTS, EACH MEASURED RATHER THAN ASSUMED, AND EACH A REAL HOLE:
#
#   - A path whose PARENT DIRECTORY does not exist here is not a claim about
#     this tree. docs/MASTERPLAN.md names src/lowlevel/gdeflateKernels.cu and
#     src/shader/gdeflate_decompress.cu, and docs/BENCHMARK-METHODOLOGY.md
#     names src/lowlevel/LZ4CompressionKernels.cu; all three live in another
#     project's tree and are separable without an exemption list because
#     src/lowlevel/ and src/shader/ do not exist here. The hole: a document
#     naming src/newdir/foo.cu, where the whole directory is owed, passes.
#   - The extraction is anchored on top-level directories and file extensions
#     DERIVED from the tracked tree, so a path under a directory or with an
#     extension this tree has never had is not matched at all. Same shape,
#     same hole. Derived rather than listed because a hand-written list goes
#     stale silently in the direction that makes this check pass more.
#   - The past-tense registers are out of scope, named below with the reason.
#     Measured: docs/BENCHMARKS.md line 473 names src/lz4_decode.cuh, which
#     existed when that measurement was written and was renamed to
#     src/chunk_decode.cuh by 7e5949c (#150). That register is append-only, so
#     the entry is honest text and the repair this check asks for - correct
#     the sentence - is not available there.
#
# ONE TRAP, FOUND BY WALKING INTO IT WHILE WRITING THIS CHECK'S OWN
# DOCUMENTATION: a document illustrating the declaration spelling with a real
# path and a real issue number IS a live declaration, judged like any other, so
# it reds the day that issue closes. CONTRIBUTING.md therefore shows the shape
# with a placeholder under a directory this tree does not have, and says why.
#
# The tracker is consulted ONLY when a declaration exists, so a tree carrying
# none needs no network and no credential. Where one does exist and the
# tracker cannot be read, this is a hard error and never a skip: a check that
# could not evaluate reads exactly like a clean one.
set -eu

root="${1:-.}"

die() {
    echo "check-doc-paths: $1" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || die "git is not on PATH"

# The past-tense registers, excluded by name with the reason each one is out.
# They are append-only records of what was true when the entry was written, so
# a path that has since been renamed is honest text there.
is_excluded() {
    case "$1" in
    CHANGELOG.md | docs/BENCHMARKS.md) return 0 ;;
    *) return 1 ;;
    esac
}

alternation_of_top_dirs() {
    git -C "$1" ls-files |
        grep '/' |
        cut -d/ -f1 |
        sort -u |
        sed 's/\./\\./g' |
        paste -sd'|' -
}

alternation_of_extensions() {
    git -C "$1" ls-files |
        sed -n 's/.*\.\([A-Za-z0-9][A-Za-z0-9]*\)$/\1/p' |
        sort -u |
        paste -sd'|' -
}

# $1 root, $2 the name of a function mapping an issue number to its state.
# The state lookup is a parameter so the self-tests below drive this exact
# code with a stub instead of the tracker. There is no environment override,
# so nothing outside this file can substitute one into a real run.
scan() {
    scan_root="$1"
    state_fn="$2"

    dirs="$(alternation_of_top_dirs "$scan_root")"
    exts="$(alternation_of_extensions "$scan_root")"
    [ -n "$dirs" ] || die "no tracked path under a directory in $scan_root - reading the wrong place"
    [ -n "$exts" ] || die "no tracked file with an extension in $scan_root - reading the wrong place"

    pathre="($dirs)/[A-Za-z0-9_./-]+\.($exts)"
    declre="$pathre\`? \(planned: #[0-9]+\)"

    scan_failures=0
    scan_paths=0

    for doc in $(git -C "$scan_root" ls-files '*.md'); do
        is_excluded "$doc" && continue

        # Prettier wraps prose at 80 columns, so a declaration can be split
        # across a line break. Normalising the document to one line first is
        # what keeps a wrapped declaration from reading as an undeclared path.
        norm="$(tr '\n' ' ' <"$scan_root/$doc" | tr -s ' ')"

        # Property 2: every declared forward reference, judged on its issue.
        printf '%s' "$norm" | grep -oE "$declre" >"$tmp/decls.txt" || true
        while IFS= read -r decl; do
            [ -n "$decl" ] || continue
            dpath="$(printf '%s' "$decl" | sed -E 's/`? \(planned: #[0-9]+\)$//')"
            dissue="$(printf '%s' "$decl" | sed -E 's/.*#([0-9]+)\)$/\1/')"
            dstate="$("$state_fn" "$dissue")"
            case "$dstate" in
            OPEN) ;;
            CLOSED)
                if [ -e "$scan_root/$dpath" ]; then
                    echo "FAIL: $doc declares $dpath as owed by #$dissue, which is closed, and the path exists. Delete the declaration."
                else
                    echo "FAIL: $doc declares $dpath as owed by #$dissue, which is closed, and the path does not exist. The promise outlived the work it pointed at."
                fi
                scan_failures=$((scan_failures + 1))
                ;;
            *) die "could not read the state of issue #$dissue. A declaration is judged against the tracker or not at all." ;;
            esac
        done <"$tmp/decls.txt"

        # Property 1: every path that is not part of a declaration. The sed
        # delimiter is @ rather than / or | because the pattern carries both.
        printf '%s' "$norm" |
            sed -E "s@$declre@<<declared>>@g" |
            grep -oE "$pathre" |
            sort -u >"$tmp/plain.txt" || true
        while IFS= read -r p; do
            [ -n "$p" ] || continue
            scan_paths=$((scan_paths + 1))
            [ -e "$scan_root/$p" ] && continue
            # The carve-out: a path whose parent directory this tree does not
            # have is a reference to somewhere else, not a stale reference to
            # here.
            [ -d "$scan_root/$(dirname "$p")" ] || continue
            echo "FAIL: $doc names $p, which does not exist. Correct the path, or declare it as owed: \`$p\` (planned: #NNN)."
            scan_failures=$((scan_failures + 1))
        done <"$tmp/plain.txt"
    done

    [ "$scan_paths" -gt 0 ] ||
        die "no repository path found in any tense-present document in $scan_root - this check is reading the wrong place"

    return "$scan_failures"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

tracker_issue_state() {
    command -v gh >/dev/null 2>&1 ||
        die "gh is not on PATH and a declared forward reference is judged against the tracker or not at all"
    gh issue view "$1" --json state --jq '.state' 2>/dev/null || echo UNREADABLE
}

stub_open() { echo OPEN; }
stub_closed() { echo CLOSED; }

# --- the proof that it bites ------------------------------------------------
#
# Run on every invocation, before the real tree is judged, because a check
# whose regex stopped matching passes everything silently and reads exactly
# like a clean tree. Each fixture is a near neighbour of a passing one.
#
# The fixture tree carries one file per extension the fixtures name, so a
# green verdict is never the extension carve-out standing in for the property
# under test: without src/batch.cu the foreign-tree case below would pass
# because .cu was unmatched rather than because src/lowlevel/ is absent.
fixture_root() {
    fr="$tmp/fx$1"
    rm -rf "$fr"
    mkdir -p "$fr/src/nested" "$fr/docs"
    : >"$fr/src/chunk_decode.cuh"
    : >"$fr/src/decode_sequence.h"
    : >"$fr/src/batch.cu"
    : >"$fr/docs/MASTERPLAN.md"
    git -C "$fr" init -q
    git -C "$fr" config core.autocrlf false
    printf '%s' "$fr"
}

selftest() {
    name="$1"
    want="$2"
    body="$3"
    state_fn="$4"

    fr="$(fixture_root "$name")"
    printf '%s\n' "$body" >"$fr/docs/MASTERPLAN.md"
    git -C "$fr" add -A

    got=0
    scan "$fr" "$state_fn" >"$tmp/out.$name" 2>&1 || got=$?
    if [ "$want" = red ] && [ "$got" -eq 0 ]; then
        cat "$tmp/out.$name" >&2
        die "self-test '$name' passed a tree it must refuse. This check is refusing nothing and its verdict on the real tree means nothing."
    fi
    if [ "$want" = green ] && [ "$got" -ne 0 ]; then
        cat "$tmp/out.$name" >&2
        die "self-test '$name' refused a tree it must accept."
    fi
    echo "PASS: self-test $name ($want)"
}

# Property 1 bites, in #391's own shape: that issue's correction reverted -
# the pre-rename source name put back into a present-tense sentence.
selftest stale-path red \
    'The parser reaches `src/lz4_decode.cuh` for the gather.' stub_open

# The same sentence with the correction in place is green, so the red above is
# the path and not the prose around it.
selftest current-path green \
    'The parser reaches `src/chunk_decode.cuh` for the gather.' stub_open

# The carve-out proven rather than assumed: another project's tree passes
# because its directory is absent here, not because the scan missed it.
selftest foreign-tree green \
    'nvCOMP decodes in `src/lowlevel/LZ4CompressionKernels.cu`, and this one in `src/batch.cu`.' stub_open

# A directory that DOES exist here, holding a file that does not, is the other
# side of that carve-out and must red.
selftest nested-missing red \
    'The table lives in `src/nested/tables.h`, beside `src/chunk_decode.cuh`.' stub_open

# Property 2, both directions over one fixture: a declared reference whose
# issue is open is honest design text; the same sentence once that issue has
# closed is a promise that outlived its work.
selftest declared-open green \
    'The table will live in `src/gdeflate_tables.h` (planned: #175), beside `src/chunk_decode.cuh`.' stub_open

selftest declared-closed red \
    'The table will live in `src/gdeflate_tables.h` (planned: #175), beside `src/chunk_decode.cuh`.' stub_closed

# The declaration is read across a line break, because Prettier wraps prose at
# 80 columns and a declaration the formatter split must not read as an
# undeclared path.
selftest declared-wrapped green \
    'The GDeflate canonical table will live in `src/gdeflate_tables.h`
(planned: #175), beside the existing `src/chunk_decode.cuh` gather.' stub_open

# --- the real tree ----------------------------------------------------------
failures=0
scan "$root" tracker_issue_state || failures=$?
[ "$failures" -eq 0 ] ||
    die "$failures document path reference(s) the tree refutes. Each line above names the document and the path."

echo "PASS: every path a tense-present document names resolves, and every declared forward reference names an open issue"
