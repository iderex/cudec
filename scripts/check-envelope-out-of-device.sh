#!/usr/bin/env bash
# No TileStream envelope symbol in the shipped device code (issue #177).
#
# The container is DirectStorage's, the page is the format, and the whole
# design of the GDeflate path rests on the kernel knowing only the second: the
# host validates where every page is and how many bytes it must produce, and
# the device is handed raw pages with bounds already decided. A symbol from
# src/tilestream.h inside a cubin would mean that separation had been given up,
# and nothing else in the tree would notice - the code would still build, still
# pass, and still decode.
#
# WHAT THIS PROVES, AND THE HALF IT DOES NOT. A matching symbol means the
# separation is gone. The converse does NOT follow, and the gap is not
# hypothetical: everything in src/tilestream.h is `inline`, so the way the
# envelope would actually reach device code is somebody marking the parser
# CUDEC_HOST_DEVICE and calling it from a kernel - at which point nvcc very
# likely inlines it whole and emits no symbol for this to find. None of
# src/gdeflate_block.h's host-device functions appear as symbols today for
# exactly that reason. So this is a DRIFT DETECTOR over the binary, not
# tamper-proofing and not a verification of the property. The companion is the
# configure-time source check in tests/CMakeLists.txt, which reads the device
# translation units CMake itself names and refuses an include of tilestream.h
# among them; that one covers the inlined case and this one covers the case
# where the header arrives by a route the source check cannot see. Neither
# subsumes the other.
#
# It reads the BINARY rather than the source for its own half deliberately:
# grepping src/ for an #include is the source check's job, and a check that did
# only that would pass on any route that did not go through an include.
#
# Fail-closed in every direction: a missing tool, a bad argument count, an
# object list that yields no device code, a symbol listing with no kernel in it
# at all, or a grep that could not run reds this. A vacuous run cannot read as
# a clean one - which matters more here than on a positive check, because "the
# forbidden symbol is absent" is exactly what an empty listing says.
#
# Usage: check-envelope-out-of-device.sh <cuobjdump> <object;list> [forbidden]
#   cuobjdump    the CUDA object dumper beside the nvcc that built the objects
#   object;list  the library's object files, ONE semicolon-joined argument; the
#                ones carrying device code are read, the host-only ones are
#                skipped and said so
#   forbidden    the extended regular expression a device symbol may not match;
#                defaults to the envelope's type prefix. The ctest self-proof
#                passes a pattern that IS present, so the refusal below is
#                proven to fire rather than assumed to.
set -u

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# EXACTLY two or three, never "at least". The object list arrives as one
# semicolon-joined generator expression; if it ever splits, an "at least"
# test would silently take the first object as the whole list and the second
# as the forbidden pattern, and the run would pass having checked one object
# against a filename. The sibling check pins its count for the same reason.
if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    fail "want 2 or 3 arguments: <cuobjdump> <object;list> [forbidden], got $#"
fi
cuobjdump="$1"
[ -x "$cuobjdump" ] || fail "no executable cuobjdump at $cuobjdump"
IFS=';' read -r -a objects <<< "$2"
[ "${#objects[@]}" -gt 0 ] && [ -n "${objects[0]}" ] || fail "no objects given"
forbidden="${3:-TileStream}"

# The kernel that MUST be there. It is the anti-vacuity control: if the symbol
# extraction silently produced nothing, this is missing and the run reds,
# instead of reporting the forbidden pattern absent from an empty listing.
required='gdeflate_decode_batch'

all_syms=""
device_objects=0
for obj in "${objects[@]}"; do
    [ -f "$obj" ] || fail "object does not exist: $obj"
    # A host-only object carries no ELF section for cuobjdump to read. That is
    # a skip and it is printed; the tool's own message goes to the log rather
    # than to /dev/null, so an object that DOES carry device code and failed
    # to dump is readable afterwards instead of being silently reclassified.
    if syms="$("$cuobjdump" --dump-elf-symbols "$obj" 2>&1)" && [ -n "$syms" ]; then
        device_objects=$((device_objects + 1))
        all_syms="$all_syms
$syms"
    else
        echo "skipped (no device code): $obj"
    fi
done

[ "$device_objects" -gt 0 ] || fail "no object carried device code; nothing was checked"
printf '%s\n' "$all_syms" | grep -q "$required" ||
    fail "no $required symbol in any device object; the listing is not the one this check is about"

# grep's exit status is read rather than discarded with `|| true`. 0 is a hit,
# 1 is a clean miss, and anything else is grep saying it could not evaluate the
# pattern at all - a bad regex reaching this from the ctest argument would
# otherwise be indistinguishable from a clean run and would PASS.
hits="$(printf '%s\n' "$all_syms" | grep -E "$forbidden")"
grep_status=$?
if [ "$grep_status" -gt 1 ]; then
    fail "grep could not evaluate the pattern '$forbidden' (exit $grep_status); nothing was checked"
fi
if [ "$grep_status" -eq 0 ]; then
    printf '%s\n' "$hits" | head -20 >&2
    fail "a device symbol matches '$forbidden'; the envelope has reached device code"
fi
echo "PASS: $device_objects device object(s) carry $required and no symbol matching '$forbidden'"
