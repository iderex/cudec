#!/usr/bin/env bash
# The TileStream envelope must not reach device code (issue #177).
#
# The container is DirectStorage's, the page is the format, and the whole
# design of the GDeflate path rests on the kernel knowing only the second: the
# host validates where every page is and how many bytes it must produce, and
# the device is handed raw pages with bounds already decided. A symbol from
# src/tilestream.h appearing inside a cubin would mean that separation had been
# given up somewhere, and nothing else in the tree would notice - the code
# would still build, still pass, and still decode.
#
# So this reads the BINARY, never the source. Grepping src/ for an #include
# would pass on exactly the failure this exists to catch, because the way the
# separation is lost is a header pulled into a .cu, and a check on the source
# has to guess which .cu files are device translation units.
#
# Fail-closed in every direction: a missing tool, an object that yields no
# device code, or a symbol listing with no kernel in it at all reds this. A
# vacuous run cannot read as a clean one - which matters more here than on a
# positive check, because "the forbidden symbol is absent" is exactly what an
# empty listing says.
#
# Usage: check-envelope-out-of-device.sh <cuobjdump> <object;list> [forbidden]
#   cuobjdump    the CUDA object dumper beside the nvcc that built the objects
#   object;list  the library's object files; the ones carrying device code are
#                read, the host-only ones are skipped and said so
#   forbidden    the regular expression a device symbol may not match; defaults
#                to the envelope's namespace prefix. The ctest self-proof
#                passes a pattern that IS present, so the refusal below is
#                proven to fire rather than assumed to.
set -u

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[ "$#" -ge 2 ] || fail "want at least 2 arguments: <cuobjdump> <object;list> [forbidden], got $#"
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
    # A host-only object has no ELF section for cuobjdump to read; that is a
    # skip and it is printed, never a silent pass.
    syms="$("$cuobjdump" --dump-elf-symbols "$obj" 2>/dev/null)" || syms=""
    if [ -z "$syms" ]; then
        echo "skipped (no device code): $obj"
        continue
    fi
    device_objects=$((device_objects + 1))
    all_syms="$all_syms
$syms"
done

[ "$device_objects" -gt 0 ] || fail "no object carried device code; nothing was checked"
printf '%s\n' "$all_syms" | grep -q "$required" ||
    fail "no $required symbol in any device object; the listing is not the one this check is about"

hits="$(printf '%s\n' "$all_syms" | grep -E "$forbidden" || true)"
if [ -n "$hits" ]; then
    printf '%s\n' "$hits" | head -20 >&2
    fail "a device symbol matches '$forbidden'; the envelope has reached device code"
fi
echo "PASS: $device_objects device object(s) carry $required and no symbol matching '$forbidden'"
