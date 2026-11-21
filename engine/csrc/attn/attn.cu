// Attention core of one Inkling-2-Mini layer visit for M = 1..8 consecutive decode tokens, in ONE kernel.
// Contract: engine/SPEC.md, area `attn`.
//
// Design
// ------
// * Work: one thread-block cluster of CS = 16 blocks per KV head (grid 16 x 8 = 128 blocks). The cached
//   keys [0, L_old) (full layer: positions 0..p0-1 of the linear cache; sliding layer: the 1024 ring slots
//   with their kpos) are dealt to the 16 blocks in tiles, round robin, so every block gets an equal share
//   of the expensive relative-bias tail (distance < 1024). Block 0 (the "owner") also takes the call's M
//   new keys. Everything is sized from `pos` on the device, so the launch shape is fixed for a CUDA graph.
// * At short contexts the kernel is bound by dependent round trips, not bytes, so it has few of them:
//   (1) every input that does not depend on `pos` (q / rel / gate / sink rows of `proj` and their scales,
//   the owner's conv inputs and rel_proj[:, 0..7]) is fetched with cp.async into shared memory while
//   `pos` is read: one round trip. The K/V stream is issued right after it lands (issued earlier, the
//   4 MB burst of a sliding layer delays that round trip by ~2 us). (2) The rel_proj column of each key
//   is fetched together with its K/V for the distance a full, in-order ring (or the linear cache)
//   implies, and re-read only if the actual kpos disagrees. (3) q / rel post-processing (scale, L2 clip,
//   BF16, log scaling) is done once per block by a few warps, branch-free, and shared through shared
//   memory. (4) Split partials (max, sum, unnormalised o) go to the workspace (L2); one cluster barrier;
//   block r then produces output dims 8r..8r+7 of every row (sink logit, sigmoid gate, BF16) from the 16
//   partials. (Distributed shared memory was measured slower here: ~1 GB/s per SM on this GPU.)
//   (5) Programmatic dependent launch: weights are staged before griddepcontrol.wait, and dependents are
//   released after the main loop. Everything an earlier kernel writes (proj, pos, conv states, caches,
//   kpos, workspace) is read after the wait only: cp.async / coherent `asm volatile` loads with memory
//   clobbers, which the compiler cannot hoist (checked in the SASS: only the weight cp.asyncs precede
//   the ACQBULK).
// * Main loop, M = 1: warp = key slice (8 warps). A half-warp owns one key; lane c of the half holds dims
//   8c..8c+7, so each K and V row (256 B) is read with 16-byte vector loads (coherent, no L1
//   allocation). Two register buffers of 2 key pairs per warp alternate (one in flight while the other
//   is consumed): 16-32 KB in flight per block, enough for DRAM bandwidth with 128 blocks. Scores are
//   reduced over the 16 lanes with xor shuffles; the bias rel . rel_proj[:, d] is reduced the same way
//   and rounded to BF16, and skipped warp-uniformly when no key of the warp is within 1024. Online
//   softmax in the log2 domain (q is pre-scaled by log2(e)/sqrt(128)), FP32 accumulators.
// * Main loop, M > 1, on tensor cores: the block loads each tile of 32 keys once with cp.async into
//   shared memory (own double-buffered region, the first tile issued before q / rel processing; with the
//   tile's kpos and rel_proj window). Rows r = 3 token + head (8-padded). S^T = K Q^T with mma.m16n8k16
//   bf16 (exact bf16 q, k; fp32 sums; 1/sqrt(128) applied to the sums), one (16 keys, 8 rows) item per
//   warp; mask and relative bias per element; per-item row max and P = exp2(S - max) stored as fp16
//   hi + lo (P - hi). Then O^T += V^T P^T with mma.m16n8k16 fp16 (V converted to fp16 in place, exact for
//   RMS-normed values), warp w owning dims 16w..16w+15, each item rescaled to the running row max in
//   registers. Two block barriers per tile. The owner runs the call's own keys first as a one-m-tile tile.
// * The owner runs the key/value short convolutions of its head (+ key clip / value RMS norm) into shared
//   memory; it is the only reader of those conv-state channels. Its global writes (conv states, new
//   cache rows) come at the end. Sliding layers, M = 1: the ring slot and kpos it overwrites hold the
//   entry at distance 1024 (invalid for this token either way; a kpos >= p0 is ignored too), so the
//   other heads may read either value. M > 1: the overwritten slots still hold keys earlier tokens of
//   the call must see: the owner writes them after the cluster barrier (its head is done reading) and
//   the last head to get there (global counter in the workspace, reset by itself) writes kpos.
// * Ring semantics: a cached ring entry is used iff 0 <= kpos < p0 and p - kpos < 1024; entries at
//   kpos >= p0 (left over from rejected speculative tokens) are ignored, as if overwritten first.
//   Values of masked keys are never multiplied in, so unused cache slots may hold anything.
// * FP8 full-layer caches (template F8; sliding rings are BF16 only): E4M3 without scale. The owner rounds
//   its new keys / values to BF16 and then to E4M3 (round to nearest even, saturated to +-448) in shared
//   memory, so the call's own keys are attended with exactly the values the cache will hold, and writes the
//   E4M3 bytes at the end. M = 1 reads 8-byte vectors with the same lanes, dims and buffer depth (half the
//   bytes in flight: the M = 1 kernels already use 128 registers). M > 1 copies a tile's E4M3 rows into the
//   first half of its stage, then expands them in place to the BF16 K / FP16 V layout (each thread loads the
//   chunks it copied, block barrier, writes); E4M3 values are exact in both.
// * Capturable: no allocation, no host sync, current stream; positions come from `pos` on the device.
// * Positions are int32 on the device (row offsets 64-bit); the host caps full caches at 2^30 rows.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>


namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_SLIDING = 8600, R_FULL = 8664;
#ifndef INGINE_ATTN_CS
#define INGINE_ATTN_CS 16
#endif
// blocks per KV head = cluster size; block r outputs dims [OD r, OD r + OD). 16 needs a GPC with 16 free SMs
// (RTX 5090); RTX 5080 GPCs cannot host a 16-block cluster of this kernel, so its build uses 8.
constexpr int CS = INGINE_ATTN_CS;
constexpr int OD = D / CS;
constexpr int MAXM = 8;
constexpr int U = 2;  // key pairs per warp per register buffer
constexpr int MAXWARPS = 8;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) * log2(e)
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;  // torch: x / scalar == x * (1 / float(scalar))
constexpr float INV_3 = 1.f / 3.f;
// workspace: int32 counter (kpos ordering for M > 1, zeroed once at allocation) then the split partials
// fp32 [HKV][CS][3M][PSTR]: o (D) then (max, sum)
constexpr size_t WS_PART = 256;
constexpr int PSTR = D + 4;
constexpr size_t ws_bytes(int M) { return WS_PART + size_t(HKV) * CS * GQA * M * PSTR * 4; }

constexpr int NT = 256;  // threads per block (8 warps) for every M

// dynamic shared memory layout (floats), per M
template <int M>
struct Smem {
    static constexpr int ROWS = GQA * M;
    // staged inputs (dead once the main loop starts)
    static constexpr int Q = 0;                          // proj q rows [M][3][D]
    static constexpr int QS = Q + M * GQA * D;           // q_scale [3][D]
    static constexpr int REL = QS + GQA * D;             // proj rel [M][3][16]
    static constexpr int RS = REL + M * GQA * DREL;      // r_scale [3][16]
    static constexpr int ST = RS + GQA * DREL;           // conv state [3][2][D]
    static constexpr int X = ST + 3 * 2 * D;             // k/v projections [M][2][D]
    static constexpr int W = X + M * 2 * D;              // conv weights [2][D][4] (sliding) | [2][4][D] (full)
    static constexpr int DYN = W + 2 * D * 4;            // dynamic taps [M][2][4]
    static constexpr int NKF = DYN + M * 2 * 4;          // conv outputs [2][M][D]
    static constexpr int STAGE_END = NKF + 2 * M * D;
    // block merge, aliases the staged inputs
    static constexpr int RED_O = 0;                      // [warps][3][D]
    static constexpr int RED_ML = RED_O + MAXWARPS * GQA * D;
    static constexpr int MERGE_END = RED_ML + MAXWARPS * GQA * 2;
    static constexpr int UNION_END = STAGE_END > MERGE_END ? STAGE_END : MERGE_END;
    // M > 1: K/V tiles loaded by the whole block, double-buffered (own region, so the first tile is in
    // flight while q / rel and the owner's convolutions are computed). K rows are bf16, V rows are
    // converted to fp16 in place once landed; 16-byte chunks are XOR-swizzled by key.
    static constexpr int KT = 32;                        // keys per tile
    static constexpr int KVS = (UNION_END + 3) & ~3;     // 16-bit [2][KT][2][D] (as floats: 2 KT D)
    static constexpr int KPS = KVS + (M > 1 ? 2 * KT * D : 0);  // kpos (int) [2][KT]
    static constexpr int SRP = KPS + (M > 1 ? 2 * KT : 0);      // rel_proj window [2][DREL][KT + MAXM]
    static constexpr int TILE_END = SRP + (M > 1 ? 2 * DREL * (KT + MAXM) : 0);
    // dedicated regions
    static constexpr int GATE = (TILE_END + 3) & ~3;     // this block's output gates [ROWS][OD]
    static constexpr int SINK = GATE + ROWS * OD;        // [ROWS]
    static constexpr int NKV = (SINK + ROWS + 3) & ~3;   // new k/v bf16 [2][M][D] (as floats: M * D)
    static constexpr int RELP0 = NKV + M * D;            // rel_proj[:, 0..7] [16][8]
    static constexpr int NST = RELP0 + DREL * 8;         // advanced conv state [3][2][D]
    // M > 1 tensor-core path: query rows r = 3 token + head, padded to NRP (multiple of 8)
    static constexpr int NRP = M > 1 ? (ROWS + 7) / 8 * 8 : 0;
    static constexpr int QB = NST + 3 * 2 * D;           // q bf16 [NRP][D] (swizzled; as floats NRP D / 2)
    static constexpr int RELD = QB + NRP * D / 2;        // rel fp32 [NRP][16]
    static constexpr int PSTRIDE = KT + 8;               // fp16 per P row (80 B: conflict-free ldmatrix)
    static constexpr int PS = RELD + NRP * DREL;         // P as fp16 hi, lo [2][NRP][PSTRIDE]
    static constexpr int ITM = PS + NRP * PSTRIDE;       // per key m-tile row (max, sum) float2 [KT/16][NRP]
    static constexpr int END = ITM + 2 * (KT / 16) * NRP;
    static constexpr size_t BYTES = size_t(END) * 4;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// D += A (16x16, row) * B (16x8, col), fp32 accumulators
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Params {
    const float* proj;
    int R;
    long long cap;  // cache rows (full layers)
    int log_scaling;
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;
    const float* k_conv;
    const float* v_conv;
    float* conv_k;
    float* conv_v;
    void* k_cache;  // __nv_bfloat16, or E4M3 bytes (F8)
    void* v_cache;
    long long* kpos;
    const long long* pos;
    int* counter;
    float* part;
    __nv_bfloat16* out;
};

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ void unpack8(const uint4 u, float f[8]) {
    f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16), f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16), f[7] = __uint_as_float(u.w & 0xffff0000u);
}

// E4M3 pair (low byte = lower dim) to floats, exact (cvt.f16x2.e4m3x2, then FP16 -> FP32)
__device__ __forceinline__ float2 e4m3x2_f2(uint32_t pair) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
}

// 8 E4M3 values (dims 8c..8c+7, byte 0 = lowest dim)
__device__ __forceinline__ void unpack8(const uint2 u, float f[8]) {
    float2 a = e4m3x2_f2(u.x & 0xffffu), b = e4m3x2_f2(u.x >> 16);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
    a = e4m3x2_f2(u.y & 0xffffu), b = e4m3x2_f2(u.y >> 16);
    f[4] = a.x, f[5] = a.y, f[6] = b.x, f[7] = b.y;
}

// x rounded to E4M3: nearest even, saturated to +-448 (torch: x.clamp(-448, 448).to(float8_e4m3fn))
__device__ __forceinline__ float e4m3r(float x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), __NV_E4M3)));
}

// two floats holding E4M3 values to their bytes (a in the low byte)
__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
    return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// Loads of anything an earlier kernel writes (caches, kpos, pos) are coherent `asm volatile` with a memory
// clobber, so the compiler cannot hoist them above griddepcontrol.wait (programmatic dependent launch).
__device__ __forceinline__ uint4 ld_kv(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint2 ld_kv8(const void* ptr) {
    uint2 v;
    asm volatile("ld.global.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ long long ld_s64(const long long* ptr) {
    long long v;
    asm volatile("ld.global.s64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void cp16(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp4(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

// zero-filling variants: nothing is read when !ok
__device__ __forceinline__ void cp16z(float* dst, const float* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void cp4z(float* dst, const float* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// clip factor min(1, t / (sqrt(ss) + 1e-6)) without the IEEE sqrt / div slow-path branches (which
// serialise the six reductions of the q / rel prologue); differs from torch by a couple of FP32 ulp
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

template <typename KV>  // 8 dims of a cache row: uint4 = BF16, uint2 = E4M3
struct Buf {
    KV k[U], v[U];
    int kp[U];    // key position, -1 = no key
    int dp[U];    // distance the rel_proj prefetch was made for
    float rp[U];  // rel_proj[c][dp] (0 if dp >= 1024)
};

template <int M, bool FULL, bool F8>
__global__ void __launch_bounds__(MAXWARPS * 32, 2) attn_decode_kernel(const Params p) {
    static_assert(FULL || !F8, "FP8 caches are full-layer only");
    using CT = std::conditional_t<F8, __nv_fp8_storage_t, __nv_bfloat16>;  // cache element
    using KV = std::conditional_t<F8, uint2, uint4>;                        // 8 dims of a cache row
    constexpr int KW = 8, ROWS = GQA * M;  // KW: key-slice warps of the M = 1 path
    using L = Smem<M>;
    extern __shared__ __align__(16) float sm[];
    __nv_bfloat16(*s_nkv)[M][D] = reinterpret_cast<__nv_bfloat16(*)[M][D]>(sm + L::NKV);

    const int g = blockIdx.y, rank = blockIdx.x;  // cluster = the 16 blocks of KV head g; rank = blockIdx.x
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, c = lane & 15;
    const int tok = warp / KW, kw = warp % KW;
    const bool owner = rank == 0;
    const int R = p.R;
    CT* const kc = static_cast<CT*>(p.k_cache);
    CT* const vc = static_cast<CT*>(p.v_cache);

    // ---- stage the inputs in shared memory with cp.async (one round trip, no registers held). Weights
    // first: they may be read before the previous kernel completes (programmatic dependent launch).
    for (int i = tid; i < GQA * D / 4; i += NT) cp16(sm + L::QS + 4 * i, p.q_scale + g * GQA * D + 4 * i);
    for (int i = tid; i < GQA * DREL / 4; i += NT) cp16(sm + L::RS + 4 * i, p.r_scale + g * GQA * DREL + 4 * i);
    if (owner) {
        for (int i = tid; i < 2 * D; i += NT) {  // sliding [kv][ch][4]; full [kv][lag][D]
            const int kv = i / D, q = i % D;
            const float* w = kv ? p.v_conv : p.k_conv;
            cp16(sm + L::W + 4 * i, FULL ? w + (q / (D / 4)) * CONV_C + g * D + 4 * (q % (D / 4)) : w + (g * D + q) * 4);
        }
        for (int i = tid; i < DREL * 2; i += NT)  // rel_proj[:, 0..7]: the new keys' distances
            cp16(sm + L::RELP0 + 4 * i, p.rel_proj + (i / 2) * EXT + 4 * (i % 2));
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");  // everything below may come from earlier kernels
    for (int i = tid; i < M * GQA * D / 4; i += NT)       // q rows of heads 3g..3g+2 are contiguous
        cp16(sm + L::Q + 4 * i, p.proj + size_t(i / (GQA * D / 4)) * R + g * GQA * D + 4 * (i % (GQA * D / 4)));
    for (int i = tid; i < M * GQA * DREL / 4; i += NT)
        cp16(sm + L::REL + 4 * i,
             p.proj + size_t(i / (GQA * DREL / 4)) * R + OFF_REL + g * GQA * DREL + 4 * (i % (GQA * DREL / 4)));
    for (int i = tid; i < ROWS * OD / 4; i += NT) {
        const int row = i / (OD / 4), part = i % (OD / 4);
        cp16(sm + L::GATE + 4 * i,
             p.proj + size_t(row / GQA) * R + OFF_GATE + (g * GQA + row % GQA) * D + rank * OD + 4 * part);
    }
    for (int i = tid; i < ROWS; i += NT) cp4(sm + L::SINK + i, p.proj + size_t(i / GQA) * R + OFF_SINK + g * GQA + i % GQA);
    if (owner) {
        for (int i = tid; i < 3 * 2 * D / 4; i += NT) {  // [j][kv][D]
            const int j = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(sm + L::ST + 4 * i, (kv ? p.conv_v : p.conv_k) + j * CONV_C + g * D + 4 * q);
        }
        for (int i = tid; i < M * 2 * D / 4; i += NT) {  // [m][kv][D]
            const int m = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(sm + L::X + 4 * i, p.proj + size_t(m) * R + (kv ? OFF_V : OFF_K) + g * D + 4 * q);
        }
        if constexpr (FULL) {
            for (int i = tid; i < M * 2; i += NT)
                cp16(sm + L::DYN + 4 * i, p.proj + size_t(i / 2) * R + (i % 2 ? OFF_VDYN : OFF_KDYN) + g * 4);
        }
    }

    const int p0 = int(ld_s64(p.pos));
    if (FULL && p0 + M > p.cap) return;  // out of cache capacity (uniform: no block reaches a barrier)
    const int L_old = FULL ? p0 : WIN;
    const int P = p0 + tok;

    // ---- K/V streaming: two register buffers; the first two are issued now, K/V before the rel_proj
    // columns (whose addresses wait for `pos`). Keys go to the blocks in tiles of 2 STEP, round robin, so
    // the expensive relative-bias tail (distance < 1024) is spread over the whole cluster.
    constexpr int STEP = KW * 2 * U, TILE = 2 * STEP, STRIDE = CS * TILE;
    const int c0 = rank * TILE, old_end = L_old;
    auto load_kv = [&](Buf<KV>& b, const int base) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + (kw * U + u) * 2 + half;
            b.k[u] = b.v[u] = KV{};
            b.kp[u] = -1;
            if (j < old_end) {
                const size_t off = (size_t(j) * HKV + g) * D + c * 8;
                if constexpr (F8) {
                    b.k[u] = ld_kv8(kc + off);
                    b.v[u] = ld_kv8(vc + off);
                } else {
                    b.k[u] = ld_kv(kc + off);
                    b.v[u] = ld_kv(vc + off);
                }
                b.kp[u] = FULL ? j : int(ld_s64(p.kpos + j));
            }
        }
    };
    auto load_rp = [&](Buf<KV>& b, const int base) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + (kw * U + u) * 2 + half;
            // distance if the ring is full and in order (exact for the linear cache)
            b.dp[u] = FULL ? P - j : tok + 1 + ((p0 - 1 - j) & (WIN - 1));
            b.rp[u] = j < old_end && b.dp[u] < EXT ? __ldg(p.rel_proj + c * EXT + b.dp[u]) : 0.f;
        }
    };
    auto load = [&](Buf<KV>& b, const int base) {
        load_kv(b, base);
        load_rp(b, base);
    };
    // ---- M > 1 key tiles: constants, layout and the cp.async issue of one tile (used by the M > 1 path)
    constexpr int KT = L::KT, NE = KT + M - 1, NRP = L::NRP, NTN = NRP / 8, KM = KT / 16;
    constexpr int PSTRIDE = L::PSTRIDE;
    __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(sm + L::KVS);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(sm + L::QB);
    int* kps = reinterpret_cast<int*>(sm + L::KPS);
    float* srp = sm + L::SRP;
    __half* ps = reinterpret_cast<__half*>(sm + L::PS);
    const int gid = lane >> 2, t4 = lane & 3;
    // 16-bit offset of (key, k|v, 16-byte chunk) in stage st; chunks XOR-swizzled by key
    auto kv_off = [&](int st, int key, int kv, int ch) { return ((st * KT + key) * 2 + kv) * D + ((ch ^ (key & 7)) * 8); };
    // distance of key base + KT - 1 for token 0: the rel_proj window starts there
    auto dbase_of = [&](const int base) {
        return FULL ? p0 - (base + KT - 1) : 1 + ((p0 - 1 - (base + KT - 1)) & (WIN - 1));
    };
    auto issue = [&](const int st, const int base) {
        if constexpr (F8) {  // E4M3 rows to the first half of the stage (expand_kv converts them in place)
            char* raw = reinterpret_cast<char*>(kvs + st * KT * 2 * D);
            for (int i = tid; i < KT * 2 * (D / 16); i += NT) {
                const int key = i / (2 * (D / 16)), kv = (i / (D / 16)) % 2, ch = i % (D / 16), j = base + key;
                const bool ok = j < L_old;
                cp16z(reinterpret_cast<float*>(raw + 16 * i),
                      reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 16), ok);
            }
        } else {
            for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8), j = base + key;
                const bool ok = j < L_old;
                cp16z(reinterpret_cast<float*>(kvs + kv_off(st, key, kv, ch)),
                      reinterpret_cast<const float*>((kv ? vc : kc) + (size_t(ok ? j : 0) * HKV + g) * D + ch * 8),
                      ok);
            }
        }
        if constexpr (!FULL) {
            for (int i = tid; i < KT; i += NT)  // low word of the int64 position
                cp4z(reinterpret_cast<float*>(kps + st * KT + i),
                     reinterpret_cast<const float*>(p.kpos + (base + i < L_old ? base + i : 0)), base + i < L_old);
        }
        const int dbase = dbase_of(base);
        if (dbase < EXT) {
            for (int i = tid; i < DREL * NE; i += NT) {
                const int c2 = i / NE, e = i % NE, d = dbase + e;
                const bool ok = d >= 0 && d < EXT;
                cp4z(srp + (st * DREL + c2) * (KT + MAXM) + e, p.rel_proj + c2 * EXT + (ok ? d : 0), ok);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // the staged inputs land first: issued before them, the K/V burst would delay their round trip by
    // the whole burst (measured: ~2.5 us instead of ~0.4 us for a sliding layer)
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    Buf<KV> A, B;
    if constexpr (M == 1) {
        if (c0 < old_end) load_kv(A, c0);
        if (c0 + STEP < old_end) load_kv(B, c0 + STEP);
        if (c0 < old_end) load_rp(A, c0);
        if (c0 + STEP < old_end) load_rp(B, c0 + STEP);
    } else {
        if (rank * KT < L_old) issue(0, rank * KT);  // in flight during q / rel and the owner's work
    }

    // ---- q and rel post-processing, once per block. q: one warp per row (lane = 4 dims); rel: one
    // half-warp per row. Fixed trip counts and clamped rows (no branches), so the reductions of a warp's
    // rows overlap. Results: scaled q (fp32, M = 1) or exact bf16 q (M > 1), bf16-valued rel.
    auto tau_of = [&](int t) {
        return p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)) : 1.f;
    };
    if constexpr (M == 1) {  // 6 items, one per warp
        if (warp < 2 * GQA) {
            const int i = warp % GQA;
            const float tau = tau_of(0);
            if (warp < GQA) {
                float4* qa = reinterpret_cast<float4*>(sm + L::Q + i * D) + lane;
                const float4 a = *qa, s = reinterpret_cast<const float4*>(sm + L::QS + i * D)[lane];
                float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
                const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    x[e] = bf16r(x[e] * f);
                    if (p.log_scaling) x[e] = bf16r(x[e] * tau);
                }
                *qa = make_float4(x[0] * QSCALE, x[1] * QSCALE, x[2] * QSCALE, x[3] * QSCALE);
            } else {
                float* ra = sm + L::REL + i * DREL + c;
                const float r = *ra * sm[L::RS + i * DREL + c];
                float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
                if (p.log_scaling) y = bf16r(y * tau);
                __syncwarp();
                if (half == 0) *ra = y;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < (ROWS + 7) / 8; ++k) {
            const int rq = warp + 8 * k, row = min(rq, ROWS - 1), i = row % GQA;
            const float tau = tau_of(row / GQA);
            const float4 a = reinterpret_cast<const float4*>(sm + L::Q + row * D)[lane];
            const float4 s = reinterpret_cast<const float4*>(sm + L::QS + i * D)[lane];
            float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
            const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[e] = bf16r(x[e] * f);
                if (p.log_scaling) x[e] = bf16r(x[e] * tau);
            }
            if (rq < ROWS) {  // exact bf16 q for the tensor cores (scale applied to the scores)
                __nv_bfloat16* qb = reinterpret_cast<__nv_bfloat16*>(sm + L::QB);
                reinterpret_cast<uint2*>(qb + row * D + (((lane >> 1) ^ (row & 7)) * 8) + (lane & 1) * 4)[0] =
                    make_uint2(pack2(x[0], x[1]), pack2(x[2], x[3]));
            }
        }
#pragma unroll
        for (int k = 0; k < (ROWS + 15) / 16; ++k) {
            const int rr = warp * 2 + half + 16 * k, row = min(rr, ROWS - 1), i = row % GQA;
            const float r = sm[L::REL + row * DREL + c] * sm[L::RS + i * DREL + c];
            float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
            if (p.log_scaling) y = bf16r(y * tau_of(row / GQA));
            if (rr < ROWS) sm[L::RELD + row * DREL + c] = y;
        }
    }
    if constexpr (M > 1) {  // padding rows: zero q and rel
        for (int i = tid; i < (L::NRP - ROWS) * D / 2; i += NT) sm[L::QB + ROWS * D / 2 + i] = 0.f;
        for (int i = tid; i < (L::NRP - ROWS) * DREL; i += NT) sm[L::RELD + ROWS * DREL + i] = 0.f;
    }
    __syncthreads();
    float qv[GQA][8], relv[GQA];
    if constexpr (M == 1) {
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            const float4* qa = reinterpret_cast<const float4*>(sm + L::Q + (tok * GQA + i) * D + c * 8);
            const float4 a0 = qa[0], a1 = qa[1];
            qv[i][0] = a0.x, qv[i][1] = a0.y, qv[i][2] = a0.z, qv[i][3] = a0.w;
            qv[i][4] = a1.x, qv[i][5] = a1.y, qv[i][6] = a1.z, qv[i][7] = a1.w;
            relv[i] = sm[L::REL + (tok * GQA + i) * DREL + c];
        }
    }

    // ---- owner: new keys / values of this KV head
    if (owner) {
        float* nkf = sm + L::NKF;  // [2][M][D]
        for (int idx = tid; idx < 2 * D; idx += NT) {
            const int kv = idx / D, ch = idx % D;
            float e[3 + M];  // [state, x_0..x_{M-1}]
#pragma unroll
            for (int j = 0; j < 3; ++j) e[j] = sm[L::ST + (j * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) e[3 + m] = sm[L::X + (m * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float y;
                if constexpr (FULL) {  // dynamic conv, weights by lag (0 = current), / sqrt(9)
                    float o = 0.f;
#pragma unroll
                    for (int lag = 0; lag < 4; ++lag) {
                        const float w = __fadd_rn(sm[L::W + (kv * 4 + lag) * D + ch], sm[L::DYN + (m * 2 + kv) * 4 + lag]);
                        o = __fadd_rn(o, __fmul_rn(w, e[m + 3 - lag]));
                    }
                    y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_3);
                } else {  // static conv, conv1d order (tap 3 = current), / sqrt(5)
                    const float* w = sm + L::W + (kv * D + ch) * 4;
                    float o = __fmul_rn(w[0], e[m]);
#pragma unroll
                    for (int j = 1; j < 4; ++j) o = __fadd_rn(o, __fmul_rn(w[j], e[m + j]));
                    y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_SQRT5);
                }
                nkf[(kv * M + m) * D + ch] = bf16r(y);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i) sm[L::NST + (i * 2 + kv) * D + ch] = e[M + i];
        }
        __syncthreads();
        for (int pr = warp; pr < 2 * M; pr += NT / 32) {
            const int kv = pr / M, m = pr % M;
            const float4 x = reinterpret_cast<const float4*>(nkf + pr * D)[lane];
            const float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
            const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
            float y[4] = {x.x * f, x.y * f, x.z * f, x.w * f};
            if constexpr (F8) {  // the values the cache will hold: BF16, then E4M3 (exact in BF16)
#pragma unroll
                for (int e = 0; e < 4; ++e) y[e] = e4m3r(bf16r(y[e]));
            }
            const uint2 v = make_uint2(pack2(y[0], y[1]), pack2(y[2], y[3]));
            reinterpret_cast<uint2*>(&s_nkv[kv][m][0])[lane] = v;
        }
        __syncthreads();
    }

    // ---- attention over this block's keys
    float mrun[GQA], lrun[GQA], acc[GQA][8];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
        mrun[i] = NEG_INIT, lrun[i] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    }
    auto compute = [&](const auto& b, const bool is_new) {  // Buf<uint4> (BF16) or Buf<uint2> (E4M3)
        float s[U][GQA];
        bool valid[U], need[U];
        int dist[U];
        bool any_need = false;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kp = b.kp[u], d = P - kp;
            // kp = -1 marks a missing key; old entries need kp < p0 (and the window on sliding layers)
            const bool ok = kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
            valid[u] = ok;
            need[u] = ok && d < EXT;
            dist[u] = d;
            any_need |= need[u];
            float kf[8];
            unpack8(b.k[u], kf);
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                float t = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
                s[u][i] = hsum16(t);
            }
        }
        if (__any_sync(0xffffffffu, any_need)) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                float rp = b.rp[u];
                if (need[u] && dist[u] != b.dp[u])
                    rp = is_new ? sm[L::RELP0 + c * 8 + dist[u]] : __ldg(p.rel_proj + c * EXT + dist[u]);
#pragma unroll
                for (int i = 0; i < GQA; ++i) {
                    const float bias = hsum16(relv[i] * rp);
                    if (need[u]) s[u][i] += bf16r(bias) * LOG2E;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            float mx = mrun[i];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!valid[u]) s[u][i] = -INFINITY;
                mx = fmaxf(mx, s[u][i]);
            }
            const float sc = exp2f(mrun[i] - mx);
            mrun[i] = mx;
            lrun[i] *= sc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float vf[8];
            // a masked key's value may be anything (an unused cache slot): keep NaN/Inf out of 0 * v
            using V = std::decay_t<decltype(b.v[0])>;
            unpack8(valid[u] ? b.v[u] : V{}, vf);
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                const float pu = exp2f(s[u][i] - mrun[i]);
                lrun[i] += pu;
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
            }
        }
    };

    if constexpr (M == 1) {
        if (owner) {  // the call's own key first, from shared memory, while the K/V stream is in flight
            Buf<uint4> N;  // BF16 (holding E4M3 values when F8)
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int m2 = (kw * U + u) * 2 + half;
                N.k[u] = N.v[u] = make_uint4(0u, 0u, 0u, 0u);
                N.kp[u] = -1, N.dp[u] = -1, N.rp[u] = 0.f;
                if (m2 < M) {
                    N.k[u] = *reinterpret_cast<const uint4*>(&s_nkv[0][m2][c * 8]);
                    N.v[u] = *reinterpret_cast<const uint4*>(&s_nkv[1][m2][c * 8]);
                    N.kp[u] = p0 + m2;
                }
            }
            compute(N, true);
        }
        for (int base = c0; base < old_end; base += STRIDE) {
            compute(A, false);
            if (base + STRIDE < old_end) load(A, base + STRIDE);
            if (base + STEP < old_end) {
                compute(B, false);
                if (base + STRIDE + STEP < old_end) load(B, base + STRIDE + STEP);
            }
        }

        // the next kernel may be scheduled now; it must still griddepcontrol.wait for this grid's results
        asm volatile("griddepcontrol.launch_dependents;");

        // ---- block partial: merge the two half-warps, then the 8 warps
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            const float mo = __shfl_xor_sync(0xffffffffu, mrun[i], 16);
            const float lo = __shfl_xor_sync(0xffffffffu, lrun[i], 16);
            const float mx = fmaxf(mrun[i], mo);
            const float f1 = exp2f(mrun[i] - mx), f2 = exp2f(mo - mx);
            lrun[i] = lrun[i] * f1 + lo * f2;
#pragma unroll
            for (int e = 0; e < 8; ++e)
                acc[i][e] = acc[i][e] * f1 + __shfl_xor_sync(0xffffffffu, acc[i][e], 16) * f2;
            mrun[i] = mx;
        }
        float* red_o = sm + L::RED_O;
        float* red_ml = sm + L::RED_ML;
        __syncthreads();  // the staged inputs (aliased by red_*) are consumed
        if (half == 0) {
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                float4* dst = reinterpret_cast<float4*>(red_o + (warp * GQA + i) * D + c * 8);
                dst[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                dst[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
                if (c == 0) red_ml[(warp * GQA + i) * 2] = mrun[i], red_ml[(warp * GQA + i) * 2 + 1] = lrun[i];
            }
        }
        __syncthreads();
        // this block's partial (unnormalised o, then max and sum per row) to the workspace (L2)
        float* part = p.part + size_t(g * CS + rank) * ROWS * PSTR;
        for (int idx = tid; idx < ROWS * (D / 4); idx += NT) {
            const int row = idx / (D / 4), d4 = idx % (D / 4);
            float mx = NEG_INIT;
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_ml[(k * GQA + row) * 2]);
            float l = 0.f;
            float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int k = 0; k < KW; ++k) {
                const int w = k * GQA + row;
                const float f = exp2f(red_ml[w * 2] - mx);
                l += red_ml[w * 2 + 1] * f;
                const float4 a = reinterpret_cast<const float4*>(red_o + w * D)[d4];
                o.x += a.x * f, o.y += a.y * f, o.z += a.z * f, o.w += a.w * f;
            }
            __stcg(reinterpret_cast<float4*>(part + row * PSTR) + d4, o);
            if (d4 == 0) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mx, l));
        }
    } else {
        // ---- M > 1 on tensor cores. Rows r = 3 token + head (NRP padded), tiles of KT keys loaded once
        // per block with cp.async (double-buffered, round robin over the cluster, with kpos and the
        // rel_proj window of the distances a full in-order ring / the linear cache implies).
        //   S^T (keys x rows) = K Q^T: mma m16n8k16 bf16 (exact bf16 q and k, fp32 sums), items (16 keys,
        //   8 rows) over the warps; mask, relative bias and the tile row max (shared memory).
        //   P = exp2(S - m) as two fp16 terms hi + lo (P - hi), so P V carries ~22 bits of P; V converted
        //   to fp16 in place (exact for these values: |v| <= sqrt(128), tiny ones lose < 3e-8).
        //   O^T (dims x rows) += V^T P^T: mma m16n8k16 fp16, warp w owns dims 16w..16w+15 of every row.
        // bf16 / e4m3 values -> fp16; anything not finite and in range (only possible in masked, unused slots) -> 0
        auto half2_of = [](float a, float b) {
            a = fabsf(a) <= 65504.f ? a : 0.f;
            b = fabsf(b) <= 65504.f ? b : 0.f;
            const __half2 h = __floats2half2_rn(a, b);
            return *reinterpret_cast<const uint32_t*>(&h);
        };
        auto to_half = [&](uint32_t w) { return half2_of(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
        // V rows of stage st to fp16: each thread converts the chunks it copied (its cp.async are done)
        auto convert_v = [&](const int st) {
            for (int i = tid; i < KT * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8);
                if (kv) {
                    uint4* q = reinterpret_cast<uint4*>(kvs + kv_off(st, key, 1, ch));
                    const uint4 u = *q;
                    *q = make_uint4(to_half(u.x), to_half(u.y), to_half(u.z), to_half(u.w));
                }
            }
        };
        // F8: the E4M3 rows of stage st (first half of the stage) to the BF16 K / FP16 V layout, in place
        auto expand_kv = [&](const int st) {
            constexpr int NCH = KT * 2 * (D / 16) / NT;  // 16-byte E4M3 chunks per thread
            static_assert(NCH * NT == KT * 2 * (D / 16), "chunks must divide over the block");
            const uint4* raw = reinterpret_cast<const uint4*>(kvs + st * KT * 2 * D);
            uint4 r[NCH];
#pragma unroll
            for (int n = 0; n < NCH; ++n) r[n] = raw[tid + n * NT];  // the chunks this thread copied
            __syncthreads();  // every chunk is in registers before the stage is overwritten
#pragma unroll
            for (int n = 0; n < NCH; ++n) {
                const int i = tid + n * NT, key = i / (2 * (D / 16)), kv = (i / (D / 16)) % 2, ch = i % (D / 16);
                const uint32_t w[4] = {r[n].x, r[n].y, r[n].z, r[n].w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // dims 16 ch + 8 h .. + 7: one 16-byte chunk of the row
                    uint32_t o[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = e4m3x2_f2((w[2 * h + e / 2] >> (16 * (e % 2))) & 0xffffu);
                        o[e] = kv ? half2_of(f.x, f.y) : pack2(f.x, f.y);
                    }
                    *reinterpret_cast<uint4*>(kvs + kv_off(st, key, kv, 2 * ch + h)) = make_uint4(o[0], o[1], o[2], o[3]);
                }
            }
        };

        // this warp's S item (16 keys of the tile, 8 rows) and the rel rows it needs for the bias
        const bool qk = warp < KM * NTN;
        const int km = warp % KM, nn = warp / KM, r0 = nn * 8 + 2 * t4;
        float rel0[DREL], rel1[DREL];
#pragma unroll
        for (int k = 0; k < DREL; ++k)
            rel0[k] = qk ? sm[L::RELD + r0 * DREL + k] : 0.f, rel1[k] = qk ? sm[L::RELD + (r0 + 1) * DREL + k] : 0.f;
        // this thread's output rows n*8 + 2 t4 (+1), dims 16 warp + gid (+8): accumulators and softmax state
        // (every warp tracks the same row statistics, so no shared running max is needed)
        float oacc[NTN][4], mrow[NTN][2], lrow[NTN][2];
#pragma unroll
        for (int n = 0; n < NTN; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) oacc[n][v] = 0.f;
            mrow[n][0] = mrow[n][1] = NEG_INIT, lrow[n][0] = lrow[n][1] = 0.f;
        }
        float2* itm = reinterpret_cast<float2*>(sm + L::ITM);  // per (key m-tile, row): item max, item sum

        // Two barriers per tile: [S, item max, P relative to it, item sums] | [P V, rescaled per item]. The
        // caller's barrier before the next tile protects P, itm and the stage.
        auto consume = [&](const int st, const int base, const bool is_new) {
            const int nk = is_new ? 1 : KM;  // key m-tiles in use (the call's own keys fit in one)
            if (qk && km < nk) {
                float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int ks = 0; ks < D / 16; ++ks) {
                    uint32_t af[4], bfr[2];
                    {
                        const int q = lane >> 3, key = km * 16 + (q & 1) * 8 + (lane & 7);
                        ldsm_x4(smem_u32(kvs + kv_off(st, key, 0, ks * 2 + (q >> 1))), af);
                    }
                    {
                        const int row = nn * 8 + (lane & 7), ch = ks * 2 + ((lane >> 3) & 1);
                        ldsm_x2(smem_u32(qb + row * D + ((ch ^ (row & 7)) * 8)), bfr);
                    }
                    mma_bf16(a4, af, bfr);
                }
                const int dbase = dbase_of(base);
                const bool staged = !is_new && dbase < EXT;
                float sv[4];
#pragma unroll
                for (int v = 0; v < 4; ++v) {  // (key gid | gid + 8, row 2 t4 | 2 t4 + 1)
                    const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1), t = row / GQA, j = base + key;
                    const int kp = is_new ? (key < M ? p0 + key : -1) : (j < L_old ? (FULL ? j : kps[st * KT + key]) : -1);
                    const int d = p0 + t - kp;
                    const bool ok = row < ROWS && kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
                    float s = a4[v] * QSCALE;
                    if (ok && d < EXT) {  // bias bf16(rel[row] . rel_proj[:, d])
                        const float* rl = (v & 1) ? rel1 : rel0;
                        const int e = d - dbase;
                        float b = 0.f;
                        if (is_new) {
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], sm[L::RELP0 + k * 8 + d], b);
                        } else if (staged && e >= 0 && e < NE) {
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], srp[(st * DREL + k) * (KT + MAXM) + e], b);
                        } else {  // ring wrap point / out-of-order ring: read the column
#pragma unroll
                            for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), b);
                        }
                        s += bf16r(b) * LOG2E;
                    }
                    sv[v] = ok ? s : -INFINITY;
                }
                float mx0 = fmaxf(sv[0], sv[2]), mx1 = fmaxf(sv[1], sv[3]);
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
                    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
                }
                mx0 = fmaxf(mx0, NEG_INIT), mx1 = fmaxf(mx1, NEG_INIT);  // fully masked item: P = 0, not NaN
                float s0 = 0.f, s1 = 0.f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {  // P as fp16 hi + lo
                    const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1);
                    const float pf = exp2f(sv[v] - ((v & 1) ? mx1 : mx0));
                    const __half h = __float2half_rn(pf);
                    ps[row * PSTRIDE + key] = h;
                    ps[(NRP + row) * PSTRIDE + key] = __float2half_rn(pf - __half2float(h));
                    ((v & 1) ? s1 : s0) += pf;
                }
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    s0 += __shfl_xor_sync(0xffffffffu, s0, o);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
                }
                if (gid == 0) itm[km * NRP + r0] = make_float2(mx0, s0), itm[km * NRP + r0 + 1] = make_float2(mx1, s1);
            }
            __syncthreads();
            float f[KM][NTN][2];  // per item weight exp2(item max - new max) of each of this thread's rows
#pragma unroll
            for (int n = 0; n < NTN; ++n) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = n * 8 + 2 * t4 + h;
                    float2 it[KM];
                    float mn = mrow[n][h];
#pragma unroll
                    for (int k = 0; k < KM; ++k) {
                        it[k] = k < nk ? itm[k * NRP + r] : make_float2(NEG_INIT, 0.f);
                        mn = fmaxf(mn, it[k].x);
                    }
                    const float sc = exp2f(mrow[n][h] - mn);
                    float l = lrow[n][h] * sc;
#pragma unroll
                    for (int k = 0; k < KM; ++k) {
                        f[k][n][h] = exp2f(it[k].x - mn);
                        l += it[k].y * f[k][n][h];
                    }
                    oacc[n][h] *= sc, oacc[n][h + 2] *= sc;
                    mrow[n][h] = mn, lrow[n][h] = l;
                }
            }
#pragma unroll
            for (int ks = 0; ks < KM; ++ks) {
                if (ks >= nk) break;
                uint32_t af[4];
                {
                    const int q = lane >> 3, key = ks * 16 + (q >> 1) * 8 + (lane & 7);
                    ldsm_x4_t(smem_u32(kvs + kv_off(st, key, 1, warp * 2 + (q & 1))), af);
                }
#pragma unroll
                for (int n = 0; n < NTN; ++n) {
                    uint32_t bfr[2];
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    const __half* pr = ps + (n * 8 + (lane & 7)) * PSTRIDE + ks * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x2(smem_u32(pr), bfr);
                    mma_f16(t, af, bfr);
                    ldsm_x2(smem_u32(pr + NRP * PSTRIDE), bfr);
                    mma_f16(t, af, bfr);
#pragma unroll
                    for (int v = 0; v < 4; ++v) oacc[n][v] = fmaf(f[ks][n][v & 1], t[v], oacc[n][v]);
                }
            }
        };

        if (owner) {  // the call's own keys first, in stage 1, while tile 0 is in flight
            for (int i = tid; i < 16 * 2 * (D / 8); i += NT) {
                const int key = i / (2 * (D / 8)), kv = (i / (D / 8)) % 2, ch = i % (D / 8);
                uint4 u = make_uint4(0u, 0u, 0u, 0u);
                if (key < M) u = reinterpret_cast<const uint4*>(&s_nkv[kv][key][0])[ch];
                if (kv) u = make_uint4(to_half(u.x), to_half(u.y), to_half(u.z), to_half(u.w));
                *reinterpret_cast<uint4*>(kvs + kv_off(1, key, kv, ch)) = u;
            }
            __syncthreads();
            consume(1, 0, true);
        }
        int st = 0;
        for (int base = rank * KT; base < L_old; base += CS * KT, st ^= 1) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");  // tile `base` (the only one in flight)
            if constexpr (F8) expand_kv(st); else convert_v(st);
            __syncthreads();  // tile landed for everyone; the previous tile's P V is done
            if (base + CS * KT < L_old) issue(st ^ 1, base + CS * KT);
            consume(st, base, false);
        }

        // the next kernel may be scheduled now; it must still griddepcontrol.wait for this grid's results
        asm volatile("griddepcontrol.launch_dependents;");

        // this block's partial (unnormalised o, then max and sum per row) to the workspace (L2)
        float* part = p.part + size_t(g * CS + rank) * ROWS * PSTR;
#pragma unroll
        for (int n = 0; n < NTN; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int row = n * 8 + 2 * t4 + (v & 1), dim = warp * 16 + gid + (v >> 1) * 8;
                if (row < ROWS) __stcg(part + row * PSTR + dim, oacc[n][v]);
            }
            if (warp == 0 && gid == 0) {
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const int row = n * 8 + 2 * t4 + v;
                    if (row < ROWS) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mrow[n][v], lrow[n][v]));
                }
            }
        }
    }
    // every partial of the head is written, and every block of the head is done reading the ring
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");

    // block `rank` outputs dims 8 rank .. 8 rank + 7 of every row: thread (row, dim) = (tid / 8, tid % 8)
    for (int it = tid; it < ROWS * OD; it += NT) {  // ROWS * OD > NT when CS = 8 and M > 5
        const int row = it / OD, dim = rank * OD + it % OD;
        const float* src = p.part + size_t(g * CS) * ROWS * PSTR + row * PSTR;
        float2 ml[CS];
        float o[CS];
#pragma unroll
        for (int q = 0; q < CS; ++q) {
            ml[q] = __ldcg(reinterpret_cast<const float2*>(src + q * ROWS * PSTR + D));
            o[q] = __ldcg(src + q * ROWS * PSTR + dim);
        }
        const float sink = sm[L::SINK + row] * LOG2E;
        float mx = sink;
#pragma unroll
        for (int q = 0; q < CS; ++q) mx = fmaxf(mx, ml[q].x);
        float l = exp2f(sink - mx), a = 0.f;
#pragma unroll
        for (int q = 0; q < CS; ++q) {
            const float f = exp2f(ml[q].x - mx);
            l += ml[q].y * f;
            a += o[q] * f;
        }
        const float gate = 1.f / (1.f + expf(-sm[L::GATE + it]));
        p.out[size_t(row / GQA) * HQ * D + (g * GQA + row % GQA) * D + dim] = __float2bfloat16_rn(bf16r(a / l) * gate);
    }

    // owner: the global writes, at the end so they delay nothing (and, for M > 1 on a ring, because the
    // overwritten slots were still being read)
    if (owner) {
        for (int idx = tid; idx < 2 * M * 16; idx += NT) {  // new K/V rows
            const int kv = idx / (M * 16), m = (idx / 16) % M, c8 = idx % 16;
            const int row = FULL ? p0 + m : (p0 + m) & (WIN - 1);
            const uint4 u = reinterpret_cast<const uint4*>(&s_nkv[kv][m][0])[c8];
            CT* const dst = (kv ? vc : kc) + (size_t(row) * HKV + g) * D;
            if constexpr (F8) {  // exact: the values are E4M3 already
                float f[8];
                unpack8(u, f);
                reinterpret_cast<uint2*>(dst)[c8] = make_uint2(pack_e4m3x2(f[0], f[1]) | pack_e4m3x2(f[2], f[3]) << 16,
                                                               pack_e4m3x2(f[4], f[5]) | pack_e4m3x2(f[6], f[7]) << 16);
            } else {
                reinterpret_cast<uint4*>(dst)[c8] = u;
            }
        }
        for (int idx = tid; idx < 3 * 2 * D; idx += NT) {  // conv states of this head's channels
            const int i = idx / (2 * D), kv = (idx / D) % 2, ch = idx % D;
            (kv ? p.conv_v : p.conv_k)[i * CONV_C + g * D + ch] = sm[L::NST + idx];
        }
        if (!FULL && M == 1 && g == 0 && tid == 0) p.kpos[p0 & (WIN - 1)] = p0;
    }
    if (!FULL && M > 1 && owner && tid == 0) {  // kpos is read by every head: the last head writes it
        __threadfence();
        if (atomicAdd(p.counter, 1) == HKV - 1) {
            __threadfence();
            for (int m = 0; m < M; ++m) p.kpos[(p0 + m) & (WIN - 1)] = p0 + m;
            *p.counter = 0;
        }
    }
}

void check(const at::Tensor& t, at::ScalarType dt, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == dt, name, " has dtype ", t.scalar_type(), ", expected ", dt);
}

using KernelFn = void (*)(Params);

struct Kernel {
    KernelFn fn;
    int M;
    size_t smem;
};

template <int M, bool FULL, bool F8>
Kernel kernel_of() {
    static const Kernel k = [] {
        const KernelFn f = attn_decode_kernel<M, FULL, F8>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(Smem<M>::BYTES)));
        return Kernel{f, M, Smem<M>::BYTES};
    }();
    return k;
}

template <bool FULL, bool F8 = false>
Kernel select(int M) {
    switch (M) {
        case 1: return kernel_of<1, FULL, F8>();
        case 2: return kernel_of<2, FULL, F8>();
        case 3: return kernel_of<3, FULL, F8>();
        case 4: return kernel_of<4, FULL, F8>();
        case 5: return kernel_of<5, FULL, F8>();
        case 6: return kernel_of<6, FULL, F8>();
        case 7: return kernel_of<7, FULL, F8>();
        default: return kernel_of<8, FULL, F8>();
    }
}

// attrs[0]: cluster of CS blocks; attrs[1]: programmatic dependent launch (the kernel waits with
// griddepcontrol.wait before touching anything an earlier kernel may write; a no-op without PDL)
cudaLaunchConfig_t launch_config(const Kernel& k, cudaStream_t stream, cudaLaunchAttribute* attrs) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(CS, HKV);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = k.smem;
    cfg.stream = stream;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS, attrs[0].val.clusterDim.y = 1, attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    return cfg;
}

}  // namespace

int64_t attn_workspace_bytes(int64_t max_m) {
    TORCH_CHECK(max_m >= 1 && max_m <= MAXM, "max_m must be in 1..8");
    return int64_t(ws_bytes(int(max_m)));
}

int64_t attn_max_active_clusters(int64_t M, int64_t window) {
    TORCH_CHECK(M >= 1 && M <= MAXM, "M must be in 1..8");
    // the FP8 variants have the same launch shape, shared memory and register cap (launch bounds)
    const Kernel k = window == 0 ? select<true>(int(M)) : select<false>(int(M));
    cudaLaunchAttribute attrs[2];
    cudaLaunchConfig_t cfg = launch_config(k, nullptr, attrs);
    cfg.numAttrs = 1;  // occupancy: the cluster shape only
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, k.fn, &cfg));
    return n;
}

void attn_decode(const at::Tensor& proj, const at::Tensor& q_scale, const at::Tensor& r_scale,
                 const at::Tensor& rel_proj, const at::Tensor& k_conv, const at::Tensor& v_conv,
                 at::Tensor& conv_k_state, at::Tensor& conv_v_state, at::Tensor& k_cache, at::Tensor& v_cache,
                 at::Tensor& kpos, const at::Tensor& pos, int64_t window, bool log_scaling, at::Tensor& workspace,
                 at::Tensor& out) {
    TORCH_CHECK(window == 0 || window == WIN, "window must be 0 (full layer) or 1024 (sliding layer)");
    const bool full = window == 0;
    const auto f32 = at::kFloat, bf16 = at::kBFloat16;
    check(proj, f32, "proj");
    check(q_scale, f32, "q_scale");
    check(r_scale, f32, "r_scale");
    check(rel_proj, f32, "rel_proj");
    check(k_conv, f32, "k_conv");
    check(v_conv, f32, "v_conv");
    check(conv_k_state, f32, "conv_k_state");
    check(conv_v_state, f32, "conv_v_state");
    const auto kv_dt = k_cache.scalar_type();
    const bool f8 = kv_dt == at::kFloat8_e4m3fn;
    TORCH_CHECK(kv_dt == bf16 || f8, "k_cache has dtype ", kv_dt, ", expected BFloat16 or Float8_e4m3fn");
    check(k_cache, kv_dt, "k_cache");
    check(v_cache, kv_dt, "v_cache");  // same dtype as k_cache
    TORCH_CHECK(full || !f8, "float8_e4m3fn caches are supported on full layers (window = 0) only");
    check(pos, at::kLong, "pos");
    check(out, bf16, "out");
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(), "workspace must be a contiguous CUDA tensor");
    const int64_t M = proj.size(0);
    TORCH_CHECK(proj.dim() == 2 && M >= 1 && M <= MAXM && proj.size(1) == (full ? R_FULL : R_SLIDING),
                "proj must be [M<=8, ", full ? R_FULL : R_SLIDING, "]");
    TORCH_CHECK(q_scale.numel() == HQ * D && r_scale.numel() == HQ * DREL && rel_proj.numel() == DREL * EXT,
                "q_scale [24,128], r_scale [24,16], rel_proj [16,1024] expected");
    const std::vector<int64_t> conv_shape = full ? std::vector<int64_t>{4, CONV_C} : std::vector<int64_t>{CONV_C, 4};
    TORCH_CHECK(k_conv.sizes() == conv_shape && v_conv.sizes() == conv_shape,
                full ? "full layers: k_conv/v_conv = k_static/v_static [4, 1024] (by lag)"
                     : "sliding layers: k_conv/v_conv [1024, 4] (conv1d order)");
    TORCH_CHECK(conv_k_state.numel() == 3 * CONV_C && conv_v_state.numel() == 3 * CONV_C, "conv states [3, 1024]");
    TORCH_CHECK(k_cache.dim() == 3 && k_cache.size(1) == HKV && k_cache.size(2) == D && k_cache.sizes() == v_cache.sizes(),
                "k_cache/v_cache must be [rows, 8, 128]");
    // device positions and loop bounds are int32; 2^30 leaves headroom for the tile strides
    TORCH_CHECK(k_cache.size(0) <= (int64_t(1) << 30), "caches hold at most 2^30 positions");
    if (!full) {
        TORCH_CHECK(k_cache.size(0) == WIN, "sliding caches must have 1024 rows");
        check(kpos, at::kLong, "kpos");
        TORCH_CHECK(kpos.numel() == WIN, "kpos must be [1024]");
    }
    TORCH_CHECK(pos.numel() >= 1, "pos must hold the first position");
    TORCH_CHECK(out.numel() == M * HQ * D, "out must be [M, 3072]");
    TORCH_CHECK(workspace.nbytes() >= ws_bytes(int(M)), "workspace too small: need attn_workspace_bytes(M)");

    const c10::cuda::CUDAGuard guard(proj.device());
    Params p{};
    p.proj = proj.data_ptr<float>();
    p.R = int(proj.size(1));
    p.cap = k_cache.size(0);
    p.log_scaling = log_scaling;
    p.q_scale = q_scale.data_ptr<float>();
    p.r_scale = r_scale.data_ptr<float>();
    p.rel_proj = rel_proj.data_ptr<float>();
    p.k_conv = k_conv.data_ptr<float>();
    p.v_conv = v_conv.data_ptr<float>();
    p.conv_k = conv_k_state.data_ptr<float>();
    p.conv_v = conv_v_state.data_ptr<float>();
    p.k_cache = k_cache.data_ptr();
    p.v_cache = v_cache.data_ptr();
    p.kpos = full ? nullptr : reinterpret_cast<long long*>(kpos.data_ptr<int64_t>());
    p.pos = reinterpret_cast<const long long*>(pos.data_ptr<int64_t>());
    p.counter = reinterpret_cast<int*>(workspace.data_ptr());
    p.part = reinterpret_cast<float*>(static_cast<char*>(workspace.data_ptr()) + WS_PART);
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const Kernel k = full ? (f8 ? select<true, true>(int(M)) : select<true>(int(M))) : select<false>(int(M));
    cudaLaunchAttribute attrs[2];
    const cudaLaunchConfig_t cfg = launch_config(k, at::cuda::getCurrentCUDAStream(), attrs);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k.fn, p));
}
