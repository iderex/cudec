#!/usr/bin/env bash
# Both wave-width instantiations of the chunk decoder, for both parsers, in
# every offload architecture's code object of the HIP archive (issue #212).
#
# The source can request both widths while the build emits one: a dropped
# arm, an architecture list that lost a width, a seam that compiled a launch
# out. No NVIDIA device can notice, so this reads the BINARY - the symbol
# table of each extracted code object - and never the source, where grepping
# for the template argument would pass on exactly the failure it exists to
# catch.
#
# Fail-closed in every direction: a missing tool, an object that yields no
# code object, an architecture with no code object, a listing with no kernel
# symbol at all, or one width short reds this. Nothing here reads as "no
# missing instantiations" by accident.
#
# Usage: check-hip-wave-instantiations.sh <llvm-bin-dir> <arch;list> <object;list>
#   llvm-bin-dir  the directory of the HIP compiler, which carries
#                 llvm-objdump and llvm-readelf beside it
#   arch;list     the offload architectures the build was asked for
#   object;list   the archive's object files; the ones carrying a fat binary
#                 are read, the host-only ones are skipped and said so
set -u

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[ "$#" -eq 3 ] || fail "want 3 arguments: <llvm-bin-dir> <arch;list> <object;list>, got $#"
tooldir="$1"
objdump="$tooldir/llvm-objdump"
readelf="$tooldir/llvm-readelf"
[ -x "$objdump" ] || fail "no executable llvm-objdump at $objdump"
[ -x "$readelf" ] || fail "no executable llvm-readelf at $readelf"

IFS=';' read -r -a archs <<< "$2"
[ "${#archs[@]}" -gt 0 ] && [ -n "${archs[0]}" ] || fail "no offload architectures given"
IFS=';' read -r -a objects <<< "$3"
[ "${#objects[@]}" -gt 0 ] && [ -n "${objects[0]}" ] || fail "no objects given"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# One counter per (architecture, parser, width): how many FUNC symbols named
# the instantiation across every code object of that architecture.
declare -A found
extracted_any=0
for obj in "${objects[@]}"; do
    [ -f "$obj" ] || fail "object does not exist: $obj"
    cp "$obj" "$work/" || fail "cannot copy $obj"
    base="$work/$(basename "$obj")"
    # Writes <base>.<n>.<target> beside the copy for every bundle the
    # object's fat binary carries, and names each one on stdout.
    out="$("$objdump" --offloading "$base" 2>&1)" || fail "llvm-objdump --offloading failed on $obj: $out"
    for co in "$base".*amdgcn-amd-amdhsa--*; do
        [ -f "$co" ] || continue
        extracted_any=1
        arch="${co##*--}"
        syms="$("$readelf" --symbols --wide "$co" 2>&1)" || fail "llvm-readelf failed on $co: $syms"
        kernels="$(printf '%s\n' "$syms" | grep -E 'FUNC' | grep 'chunk_decode_batch' || true)"
        [ -n "$kernels" ] || fail "$obj carries a $arch code object with no chunk_decode_batch kernel symbol at all"
        for parser in Lz4Parser SnappyParser; do
            for width in 32 64; do
                n="$(printf '%s\n' "$kernels" | grep "$parser" | grep -c "Lb0ELi${width}E")"
                key="$arch/$parser/$width"
                found["$key"]=$(( ${found["$key"]:-0} + n ))
            done
        done
    done
done
[ "$extracted_any" -eq 1 ] || fail "no object yielded a code object; nothing was checked"

status=0
for arch in "${archs[@]}"; do
    line="$arch:"
    for parser in Lz4Parser SnappyParser; do
        for width in 32 64; do
            key="$arch/$parser/$width"
            n="${found["$key"]:-0}"
            if [ "$n" -ge 1 ]; then
                line="$line $parser<$width>"
            else
                line="$line MISSING:$parser<$width>"
                status=1
            fi
        done
    done
    echo "$line"
done
[ "$status" -eq 0 ] || fail "an instantiation is missing from the device code (see the lines above)"
echo "PASS: both wave widths of both parsers are in the device code of every architecture (${archs[*]})"
