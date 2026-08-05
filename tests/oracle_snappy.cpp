/* The Snappy oracle proving itself before any Snappy decoder exists: the
 * pinned build links, its two harness-facing verdicts agree with each other
 * on a real stream, and both refuse a stream the format cannot describe. No
 * cudec code is under test here - this is the reference the M3 parity tests
 * will be held to, checked for being present and callable at all. */
#include "fixtures.h"
#include "require.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    /* Compressible, so the stream carries a copy element rather than one
     * literal run: a link that resolved the wrong symbols would not survive
     * a round trip through both element kinds. */
    std::vector<unsigned char> original;
    const std::string token = "cudec snappy oracle ";
    while (original.size() < 4096) {
        original.insert(original.end(), token.begin(), token.end());
    }

    const auto compressed = SnappyCompressBlock(original);
    REQUIRE(!compressed.empty());
    REQUIRE(compressed.size() < original.size());
    REQUIRE(SnappyOracleAccepts(compressed));

    std::vector<unsigned char> decoded;
    REQUIRE(SnappyOracleDecodes(compressed, &decoded));
    REQUIRE_CTX(decoded.size() == original.size(), "decoded %zu of %zu bytes",
                decoded.size(), original.size());
    REQUIRE(equal_bytes(decoded.data(), original.data(), decoded.size()));

    /* Truncation is the one negative this test owes: it proves the verdicts
     * are being read from snappy rather than assumed. The reject ladder and
     * the mutation corpus are their own issues. */
    std::vector<unsigned char> truncated(compressed.begin(),
                                         compressed.end() - 1);
    REQUIRE(!SnappyOracleAccepts(truncated));
    std::vector<unsigned char> truncated_out;
    REQUIRE(!SnappyOracleDecodes(truncated, &truncated_out));
    REQUIRE(truncated_out.empty());

    /* An empty buffer never reaches snappy: the wrapper refuses it so no
     * caller can hand the reference a null pointer. */
    std::vector<unsigned char> empty;
    std::vector<unsigned char> empty_out;
    REQUIRE(!SnappyOracleAccepts(empty));
    REQUIRE(!SnappyOracleDecodes(empty, &empty_out));

    std::printf("PASS: snappy oracle round-trips %zu bytes to %zu and refuses "
                "a truncated stream\n",
                original.size(), compressed.size());
    return 0;
}
