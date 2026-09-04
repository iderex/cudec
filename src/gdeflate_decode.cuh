/* The warp-per-page GDeflate decoder (issue #214): one team of 32 lanes per
 * raw page over a grid-stride loop, the 32 lanes being the format's 32
 * substreams, one table set per team in shared memory at a footprint no input
 * can grow, and the block loop, the copy engine and the failure contract
 * shared with the CPU twin rather than restated (docs/MASTERPLAN.md section
 * 13). Internal header, not part of the ABI; src/batch.cu instantiates it.
 *
 * WHAT IS DEVICE CODE HERE AND WHAT IS NOT. Nothing in this file decodes a
 * bit. The rounds are src/gdeflate_block.h's and src/gdeflate_tables.h's,
 * written once over the team contract stated at GDeflateHostTeam, and this
 * file is the second residency of that contract: a lane's bits live in this
 * thread's registers, the team's collectives are a ballot, a prefix sum and a
 * shuffle, and a refill is one ballot over the lanes that stepped with each
 * of them taking the word its rank among them names - which is exactly the
 * word the sequential cursor would have handed it, because the schedule hands
 * words out in lane order (src/gdeflate_schedule.h). So a page that decodes
 * to different bytes on the two residencies is a defect in this file, and the
 * rounds cannot be that defect: they are the same code.
 *
 * ONE TEAM PER WAVE, WHATEVER THE WAVE WIDTH. The format has 32 substreams, so
 * a wave64 device leaves its upper 32 lanes with no substream to own. Those
 * lanes run every round with nothing to do rather than being masked out of
 * the collectives: the collectives here name the whole wave, a lane that
 * dropped out of one would leave the rest waiting, and a wave64 device has
 * never executed this decoder (issue #415). Two teams per wave, each decoding
 * its own page, is a measurement for that hardware and not a change to make
 * without it.
 *
 * FAIL-CLOSED, WITH THE BOUNDS WHERE THE TWIN HAS THEM. Every write of the
 * rounds is bounded by the caller's capacity before it happens and every read
 * of the page by the caller's size, per lane, so a lane that fails on the
 * device after a lower lane already failed cannot reach memory it should not:
 * the round finishes for every lane and the lowest refusing lane's rung is
 * the team's, which is the verdict the sequential loop gives, because no
 * lane's step depends on a higher lane's. On any refusal the page's result is
 * a non-OK status with bytes_written zero, and the destination up to that
 * point is unspecified and never presented as a decode. */
#ifndef CUDEC_GDEFLATE_DECODE_CUH
#define CUDEC_GDEFLATE_DECODE_CUH

#include "chunk_decode.cuh"
#include "cudec.h"
#include "gdeflate_block.h"

#include "vendor_rt.h"

namespace cudec_detail {

/* Every lane of the wave, in the width the backend's collectives take it. The
 * name is the one the structural rule in tests/CMakeLists.txt admits as a
 * mask, so a collective here can be read by that rule as naming the whole
 * warp. */
constexpr auto kFullWarpMask = CUDEC_RT_WAVE_FULL_MASK;

/* The per-chunk status a refused page reports through the batch ABI. The
 * three rungs that refuse against the caller's capacity are the output being
 * too small for the page, which the ABI names apart from corruption so a
 * caller can retry with a larger tile; every other rung is a page this decoder
 * will not decode at any capacity. It lives beside the kernel rather than beside
 * the ladder because the ladder's headers know no ABI status, and the twins,
 * the bench and the fuzz targets that include them must not start to. */
CUDEC_HOST_DEVICE inline cudec_status GDeflateStatusFor(GDeflateReject why) {
    switch (why) {
        case kGDeflateRejectStoredPastCap:
        case kGDeflateRejectLiteralPastCap:
        case kGDeflateRejectMatchPastCap:
            return CUDEC_ERR_OUTPUT_TOO_SMALL;
        default:
            return CUDEC_ERR_CORRUPT_INPUT;
    }
}

/* One lane's residency of the schedule primitives: its own buffer and
 * occupancy in registers, and the verdict and the page's word count beside
 * them. The verdict is per thread inside a round and team-uniform after it,
 * which is what Unify below establishes. */
struct GDeflateLane {
    uint64_t bitbuf;
    uint32_t bitsleft;
    uint64_t word_count;
    bool failed;
    GDeflateReject reject;

    __device__ uint64_t& Buf() { return bitbuf; }
    __device__ const uint64_t& Buf() const { return bitbuf; }
    __device__ uint32_t& Left() { return bitsleft; }
    __device__ const uint32_t& Left() const { return bitsleft; }
};

/* One team's table set: the two live decode tables, the precode and the
 * code-length vector it expands, and the precode's own lengths. Sized by the
 * alphabets and by nothing a page says (docs/MASTERPLAN.md section 13.1). */
struct GDeflateTeamTables {
    GDeflateLitLenTable litlen;
    GDeflateDistTable dist;
    GDeflatePrecodeTable precode;
    GDeflateCodeLengths lens;
    unsigned char precode_lens[kGDeflateNumPrecodeSyms];
};

/* THE WARP RESIDENCY OF THE TEAM CONTRACT. Each method is the collective form
 * of the sequential one at GDeflateHostTeam, and the two agree on every value
 * a round can observe. The mask type is the wide one on both backends; a
 * backend whose ballot answers in the narrow width is widened, and the team
 * mask below keeps the upper lanes of a wider wave out of every answer. */
template <int WaveSize>
class GDeflateWarpTeam {
   public:
    using Mask = unsigned long long;

    __device__ GDeflateWarpTeam(GDeflateTeamTables& tables,
                                const unsigned char* page, uint32_t lane)
        : tables_(tables),
          page_(page),
          lane_(lane),
          active_(0),
          cursor_(0) {
        me_.bitbuf = 0;
        me_.bitsleft = 0;
        me_.word_count = 0;
        me_.failed = false;
        me_.reject = kGDeflateRejectNone;
        ClearCopies();
    }

    __device__ GDeflateLane& Bits() { return me_; }
    __device__ GDeflateDeferredCopy& Copy() { return copy_; }
    __device__ uint32_t Lane() const { return lane_; }
    __device__ bool Active() const { return lane_ < active_; }
    __device__ bool Failed() const { return me_.failed; }
    __device__ GDeflateReject Reject() const { return me_.reject; }
    __device__ uint64_t Cursor() const { return cursor_; }
    __device__ uint64_t WordCount() const { return me_.word_count; }
    __device__ const unsigned char* Page() const { return page_; }

    __device__ GDeflateLitLenTable& LitLen() { return tables_.litlen; }
    __device__ GDeflateDistTable& Dist() { return tables_.dist; }
    __device__ GDeflatePrecodeTable& Precode() { return tables_.precode; }
    __device__ GDeflateCodeLengths& Lens() { return tables_.lens; }
    __device__ unsigned char* PrecodeLens() { return tables_.precode_lens; }

    /* The priming round: the page's shape checked by every lane alike, then
     * lane n takes word n, which is what thirty-two sequential refills from a
     * cursor at zero hand out. The lanes of a wider wave take nothing. */
    __device__ bool Prime(uint64_t page_bytes) {
        me_.bitbuf = 0;
        me_.bitsleft = 0;
        me_.failed = false;
        me_.reject = kGDeflateRejectNone;
        GDeflateReject why = kGDeflateRejectNone;
        if (!GDeflatePageShape(page_bytes, &me_.word_count, &why)) {
            return GDeflateRefuseAs(me_, why);
        }
        if (lane_ < kGDeflateNumStreams) {
            me_.bitbuf = GDeflateWordAt(page_, lane_);
            me_.bitsleft = kGDeflateBitsPerPacket;
        }
        cursor_ = kGDeflateNumStreams;
        return true;
    }

    /* One round: every lane runs f, then the lanes that stepped and fell
     * below the watermark take the next words in lane order. The word a lane
     * takes is the cursor plus its rank among the takers, which is the
     * sequential schedule's answer read off a ballot. */
    template <class F>
    __device__ void Round(uint32_t lanes, F f) {
        active_ = lanes;
        const bool stepped = f(*this);
        const bool need = stepped && lane_ < kGDeflateNumStreams &&
                          GDeflateLaneNeedsRefill(me_);
        const Mask takers = Ballot(need);
        if (need) {
            const uint64_t word = cursor_ + __popcll(takers & LowerMask());
            GDeflateRefillLane(me_, page_, word);
        }
        cursor_ += __popcll(takers);
        Unify();
        Sync();
    }

    template <class F>
    __device__ void Lane0(F f) {
        active_ = 1;
        if (lane_ == 0) {
            f(*this);
        }
        Unify();
    }

    __device__ void EnsureLane0() {
        const bool need = lane_ == 0 && GDeflateLaneNeedsRefill(me_);
        const Mask takers = Ballot(need);
        if (need) {
            GDeflateRefillLane(me_, page_, cursor_);
        }
        cursor_ += __popcll(takers);
        Unify();
    }

    __device__ void Reset() {}

    /* One lane builds, the rest wait: the construction is a walk over the
     * alphabet that the sequential residency runs in one thread too, and
     * making it cooperative is a measurement question (#204), not a
     * correctness one. */
    template <class F>
    __device__ void Build(F f) {
        if (lane_ == 0) {
            f(*this);
        }
        Sync();
        Unify();
    }

    template <class T>
    __device__ T Broadcast(T value) const {
        return Shfl(value, 0);
    }

    /* Positions in lane order: an inclusive prefix sum over the team, this
     * lane's exclusive share added to the base, and the base advanced past
     * every lane's claim - which is `base += len` thirty-two times over. */
    __device__ uint64_t Claim(uint64_t& base, uint64_t len) {
        uint64_t sum = (lane_ < kGDeflateNumStreams) ? len : 0;
        for (uint32_t d = 1; d < kGDeflateNumStreams; d <<= 1) {
            const uint64_t below = ShflUp(sum, d);
            if (lane_ >= d) {
                sum += below;
            }
        }
        const uint64_t total = Shfl(sum, kGDeflateNumStreams - 1u);
        const uint64_t pos = base + (sum - len);
        base += total;
        return pos;
    }

    /* The lowest flagging lane of the team, or kGDeflateNoLane. Team-wide
     * rather than at-or-below, which the contract allows: every comparison
     * the rounds make against the answer reads the same either way. */
    __device__ uint32_t FirstLane(bool flag) {
        const Mask flagged = Ballot(flag) & TeamMask();
        if (flagged == 0) {
            return kGDeflateNoLane;
        }
        return static_cast<uint32_t>(__ffsll(static_cast<long long>(flagged))) -
               1u;
    }

    /* The copies of this round in lane order, each performed by the whole
     * wave: the closed-form gather dst[i] = src[i mod offset] is what
     * byte-by-byte forwards copying MEANS for an overlapping match, and the
     * order between copies is what the deferred copy needs, since a lower
     * lane's reservation of this same round may be what a higher lane's copy
     * reads (src/gdeflate_block.h). The dst pointer is deliberately not
     * __restrict__: the next copy may read what this one wrote, and the
     * __syncwarp between them is what forces the reload. */
    __device__ void CopyBytes(bool active, unsigned char* out, uint64_t dst,
                              uint64_t src, uint64_t len) {
        const Mask copying = Ballot(active) & TeamMask();
        for (uint32_t j = 0; j < kGDeflateNumStreams; j++) {
            if (((copying >> j) & 1u) == 0) {
                continue;
            }
            const uint64_t d = Shfl(dst, j);
            const uint64_t s = Shfl(src, j);
            const uint64_t n = Shfl(len, j);
            /* The distance is at most the 64 KiB tile and the length at most
             * two bytes past it (src/gdeflate_block.h), so both fit the
             * 32-bit modulo the LZ4 kernel measured as the cheaper one. The
             * distance is at least one by the schedule's tables, so the
             * modulo cannot be by zero. */
            const uint32_t dist = static_cast<uint32_t>(d - s);
            for (uint64_t i = lane_; i < n; i += static_cast<uint32_t>(WaveSize)) {
                out[d + i] = out[s + (static_cast<uint32_t>(i) % dist)];
            }
            Sync();
        }
    }

    /* The asking lanes in lane order, each one's writes visible to the next:
     * one lane at a time behind a wave barrier, which is the only thing the
     * code-length expansion's "repeat the previous length" needs. */
    template <class F>
    __device__ void Serial(bool active, F f) {
        const Mask asking = Ballot(active) & TeamMask();
        for (uint32_t j = 0; j < kGDeflateNumStreams; j++) {
            if (((asking >> j) & 1u) == 0) {
                continue;
            }
            if (lane_ == j) {
                f(*this);
            }
            Sync();
        }
    }

    __device__ uint64_t BufferedBits() const {
        uint64_t sum = (lane_ < kGDeflateNumStreams) ? me_.bitsleft : 0;
        for (uint32_t d = 1; d < static_cast<uint32_t>(WaveSize); d <<= 1) {
            sum += ShflXor(sum, d);
        }
        return sum;
    }

    __device__ void ClearCopies() {
        copy_.out_pos = 0;
        copy_.length = 0;
        copy_.pending = false;
    }

   private:
    /* The team's verdict after a round: the lowest refusing lane's rung,
     * raised on every other lane through the choke point. A lane that
     * refused on its own keeps its rung, which is the team's if it is the
     * lowest; lane 0 is therefore always carrying the team's rung, and it is
     * the lane the kernel reports from. */
    __device__ void Unify() {
        const Mask refused = Ballot(me_.failed) & TeamMask();
        if (refused == 0) {
            return;
        }
        const uint32_t first =
            static_cast<uint32_t>(__ffsll(static_cast<long long>(refused))) -
            1u;
        const uint32_t rung = Shfl(static_cast<uint32_t>(me_.reject), first);
        GDeflateRefuseAs(me_, static_cast<GDeflateReject>(rung));
    }

    __device__ Mask Ballot(bool pred) const {
        return static_cast<Mask>(__ballot_sync(kFullWarpMask, pred ? 1 : 0));
    }
    template <class T>
    __device__ T Shfl(T value, uint32_t from) const {
        return __shfl_sync(kFullWarpMask, value, static_cast<int>(from));
    }
    template <class T>
    __device__ T ShflUp(T value, uint32_t delta) const {
        return __shfl_up_sync(kFullWarpMask, value, delta);
    }
    template <class T>
    __device__ T ShflXor(T value, uint32_t lanemask) const {
        return __shfl_xor_sync(kFullWarpMask, value, static_cast<int>(lanemask));
    }
    __device__ void Sync() const { __syncwarp(); }

    /* The team's lanes within the wave: all of a 32-lane wave, the lower half
     * of a wider one. */
    __device__ static Mask TeamMask() {
        return static_cast<Mask>(-1) >>
               (static_cast<uint32_t>(WaveSize) - kGDeflateNumStreams);
    }
    __device__ Mask LowerMask() const {
        return (static_cast<Mask>(1) << lane_) - 1u;
    }

    GDeflateTeamTables& tables_;
    const unsigned char* page_;
    uint32_t lane_;
    uint32_t active_;
    uint64_t cursor_;
    GDeflateLane me_;
    GDeflateDeferredCopy copy_;
};

/* One team per page over a grid-stride loop. The geometry guards are the
 * chunk decoder's, for the reasons it gives: fewer than one whole wave in the
 * grid makes the stride zero, and a block that is not a wave multiple splits
 * a team across blocks. A block wider than the launch shape is refused as
 * well, because the shared table set is sized for that shape. */
template <int WaveSize>
__global__ void __launch_bounds__(kBlockThreadsFor<WaveSize>)
    gdeflate_decode_batch(const void* const* src_ptrs, const size_t* src_sizes,
                          void* const* dst_ptrs, const size_t* dst_caps,
                          size_t chunk_count, cudec_chunk_result* results) {
    constexpr unsigned kWaveSize = static_cast<unsigned>(WaveSize);
    __shared__ GDeflateTeamTables tables[kBlockWarps];

    const unsigned lane = threadIdx.x % kWaveSize;
    const unsigned wave_in_block = threadIdx.x / kWaveSize;
    const size_t wave_in_grid =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize;
    const size_t total_waves =
        (static_cast<size_t>(gridDim.x) * blockDim.x) / kWaveSize;
    if (total_waves == 0 || blockDim.x % kWaveSize != 0 ||
        blockDim.x > kBlockThreadsFor<WaveSize>) {
        return;
    }

    for (size_t chunk = wave_in_grid; chunk < chunk_count;
         chunk += total_waves) {
        const unsigned char* src =
            static_cast<const unsigned char*>(src_ptrs[chunk]);
        unsigned char* dst = static_cast<unsigned char*>(dst_ptrs[chunk]);
        GDeflateWarpTeam<WaveSize> team(tables[wave_in_block], src, lane);
        uint64_t out_len = 0;
        GDeflateCensus census;
        const bool ok = GDeflateDecodePageRounds(team, src_sizes[chunk], dst,
                                                 dst_caps[chunk], &out_len,
                                                 &census);
        /* The verdict is team-uniform after every round and lane 0 carries
         * the team's rung; one lane writes the 16-byte result. bytes_written
         * is set on full success only. */
        if (lane == 0) {
            results[chunk].status =
                ok ? CUDEC_OK : GDeflateStatusFor(team.Reject());
            results[chunk].reserved = 0;
            results[chunk].bytes_written = ok ? out_len : 0;
        }
        /* The next page's table build must not overtake a lane still
         * reading this page's tables. */
        __syncwarp();
    }
}

}  // namespace cudec_detail

#endif /* CUDEC_GDEFLATE_DECODE_CUH */
