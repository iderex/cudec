#!/bin/sh
# Refuses a fuzz seed corpus that has stopped being the corpus the manifest
# describes.
#
# The failure this prevents is a fuzz job that reads as green while it seeded
# nothing. libFuzzer treats an absent or empty corpus directory as a legal
# start-from-scratch run: it prints "INFO: A corpus is not provided, starting
# from an empty corpus" and exits 0 having explored almost nothing. A deleted
# seed, a directory emptied by a bad rebase, and a corpus that was never
# checked out all arrive at that same green.
#
# Four refusals, and each one is a way the corpus can rot:
#
#   - a seed whose bytes changed, or a seed named in the manifest that is gone
#     (sha256sum -c);
#   - a file present under the corpus root that the manifest does not name, so
#     a seed cannot be added without its hash;
#   - a target directory that is empty, which is the shape that goes green
#     upstream;
#   - a fuzz target declared in fuzz/CMakeLists.txt with no corpus directory at
#     all, so adding a target without seeding it reds here rather than running
#     it from nothing.
#
# The target list is DERIVED from fuzz/CMakeLists.txt rather than written here.
# A list in this file would drift against the build that decides it, and the
# drift would be silent in the safe-looking direction: the target the list
# forgot is the one that runs unseeded.
#
# The verifier proves itself on every run before it is trusted with the real
# tree, the way scripts/check-dependabot-config.sh does: it runs first over a
# copy of the corpus with one target directory emptied, and a run that ACCEPTS
# that copy is a run whose verdict on the real corpus means nothing.
set -eu

root="${1:-fuzz/corpus}"
cmakelists="${2:-fuzz/CMakeLists.txt}"
manifest_name="SHA256SUMS"

die() {
    echo "check-fuzz-corpus: $1" >&2
    exit 1
}

command -v sha256sum >/dev/null 2>&1 || die "sha256sum is not on PATH"

[ -f "$cmakelists" ] ||
    die "$cmakelists does not exist, so the target list cannot be derived.
This check refuses rather than falling back to a list of its own."

# Every cudec_add_fuzz_target(...) name, minus the _selftest twins: those share
# their sibling's corpus and own none. Anchored at the start of the line so the
# function's own definition and the prose above it cannot contribute a name.
targets="$(
    sed -n 's/^cudec_add_fuzz_target(\([A-Za-z0-9_]*\).*/\1/p' "$cmakelists" |
        grep -v '_selftest$' |
        LC_ALL=C sort -u
)"
[ -n "$targets" ] ||
    die "no cudec_add_fuzz_target(...) call was found in $cmakelists.
An empty target list would let this check pass over any corpus at all."

# The verifier, over a root that is a corpus tree. Called twice: once on a
# planted defect, once on the real thing.
verify() {
    _root="$1"
    _manifest="$_root/$manifest_name"

    [ -d "$_root" ] || {
        echo "corpus root $_root does not exist" >&2
        return 1
    }
    [ -s "$_manifest" ] || {
        echo "$_manifest is missing or empty" >&2
        return 1
    }

    # Bytes and presence. -c reds on a changed seed and on a named seed that is
    # gone; --quiet keeps a passing run to one line of this script's own.
    (cd "$_root" && sha256sum --quiet -c "$manifest_name") || {
        echo "the corpus does not match $_manifest" >&2
        return 1
    }

    # The other direction: a file the manifest does not name. Without this a
    # seed could be added with no hash, and the pinning would cover a subset of
    # the corpus while reading as if it covered all of it.
    _listed="$(sed 's/^[0-9a-f]* [ *]//' "$_manifest" | LC_ALL=C sort)"
    _present="$(
        find "$_root" -type f |
            sed "s|^$_root/||" |
            grep -v "^$manifest_name\$" |
            LC_ALL=C sort
    )"
    _unlisted="$(printf '%s\n' "$_present" | grep -vxF "$_listed" || true)"
    [ -z "$_unlisted" ] || {
        echo "under the corpus root but absent from $manifest_name:" >&2
        printf '%s\n' "$_unlisted" >&2
        return 1
    }

    # One directory per declared target, each holding at least one seed. The
    # empty directory is the case that goes green upstream, so it is named
    # separately from the missing one.
    for _t in $targets; do
        [ -d "$_root/$_t" ] || {
            echo "$cmakelists declares fuzz target '$_t' but $_root/$_t does not exist" >&2
            return 1
        }
        _count="$(find "$_root/$_t" -type f | wc -l)"
        [ "$_count" -gt 0 ] || {
            echo "$_root/$_t holds no seed - libFuzzer would start from an empty corpus and pass" >&2
            return 1
        }
    done

    return 0
}

# The self-test. One target directory is emptied in a copy, which is exactly
# the shape the whole check exists to refuse; the manifest is trimmed to match
# so the copy fails on the emptied directory rather than on a missing-file
# hash, and the leg proves the directory arm specifically.
first_target="$(printf '%s\n' "$targets" | head -n 1)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -R "$root" "$work/planted"
# Removed and recreated rather than emptied in place, so the planted defect is
# the same shape whatever state the real tree is in - including a real tree
# whose directory is already missing, where an in-place delete would abort this
# script on find's own error and the run would red for the wrong reason.
rm -rf "$work/planted/$first_target"
mkdir -p "$work/planted/$first_target"
grep -v "$first_target/" "$work/planted/$manifest_name" >"$work/trimmed" || true
mv "$work/trimmed" "$work/planted/$manifest_name"
if verify "$work/planted" >"$work/self-test.log" 2>&1; then
    cat "$work/self-test.log" >&2
    die "the self-test corpus was accepted. Its '$first_target' directory holds
no seed at all, so a check that accepts it is refusing nothing and its verdict
on the real corpus means nothing."
fi
echo "PASS: self-test - the check refuses a corpus with an empty target directory"

verify "$root" || die "$root is not the corpus $root/$manifest_name describes"
echo "PASS: $root - $(grep -c . "$root/$manifest_name") seed(s) across $(printf '%s\n' "$targets" | wc -l | tr -d ' ') target(s), all hash-pinned"
