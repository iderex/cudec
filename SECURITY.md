# Security policy

cudec decodes untrusted input by design — compressed bitstreams are classic
attack surface, and a GPU decoder that trusts its input is a vulnerability
with a very wide blast radius. The security posture is therefore central,
not incidental:

- decoding is **fail-closed**: a malformed, truncated, or hostile bitstream
  produces a defined error, never an out-of-bounds read or write and never
  partially-written output presented as success,
- every reject path is covered by an explicit negative test, and decode
  output is fuzz-diffed against the CPU reference implementations,
- all size and offset arithmetic is checked for overflow before use,
- the toolchain and all CI actions are pinned to exact hashes.

## Reporting a vulnerability

Please report vulnerabilities (e.g. a crafted bitstream causing memory
corruption) privately via
[GitHub private vulnerability reporting](../../security/advisories/new)
rather than a public issue. Reports are usually answered within a week.
