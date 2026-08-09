# Notice

This software is developed for lawful use. Operators and users are
responsible for making sure that their deployment and use comply with the
laws that apply to them, including copyright and data protection law. The
project does not endorse or support unlawful use of any kind, and nothing
in it is designed to enable such use. The license contains the full
warranty and liability disclaimer.

## Where the GDeflate verification apparatus came from

cudec distributes none of the sources below. They are fetched at configure
time by the test build, the way the LZ4, Snappy and Zstd references already
are, and the no-vendored-binaries rule keeps them out of this tree. Neither
upstream ships a NOTICE file, so Apache-2.0 section 4(d) has nothing to
propagate here; this section exists because recording where the verification
apparatus came from is worth doing on its own terms, and it is what
`docs/MASTERPLAN.md` section 11.9 decided to carry.

**The NVIDIA fork of libdeflate, branch `gdeflate`**, pinned at commit
`8ba9502fb30d2bf728592d121f0d402e40c8cb05`. It is the reference GDeflate
codec cudec's decoder is diff-tested against, and the compressor that
generates the M4 corpora. Its licensing is mixed per file rather than
uniform: the repository's `COPYING` is Eric Biggers' MIT text from upstream
libdeflate, and `lib/gdeflate_compress.c` and `lib/gdeflate_decompress.c`
each carry that MIT notice together with an `SPDX-License-Identifier:
Apache-2.0` line and an NVIDIA copyright.

**The GDeflate subtree of `microsoft/DirectStorage`**, which carries its own
Apache-2.0 `LICENSE` with NVIDIA and Microsoft copyrights, and whose
`GDeflateTest` vectors are the one piece of interop evidence in M4 that comes
from a different implementation lineage than the fork above.

**The IETF draft `draft-uralsky-gdeflate-00`**, "GDEFLATE bitstream format
specification", used under the BCP 78 terms it is published with. It is one
of the three artifacts the format dossier in `docs/MASTERPLAN.md` section 11
is assembled from, because no canonical GDeflate specification exists.
