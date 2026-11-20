// src/kernels/cuda/i8_mma.cuh - a group of quantized matrices times rows of int8 activations on the int8 tensor cores
// (gemm_iq_i8_grouped, declared in include/strata/kernels/iq_kernels.hpp), for a matrix of any size.  Included once by
// iq_kernels.cu inside its anonymous namespace, after i8_expert.cuh: a weight sub-block is decoded by that file's
// i8_load / i8_pack / i8_convert (32 values as 8 words of int8, their integer scales and float base), Q3_K by the
// decoder here.  CUDA only, Ampere and newer: mma.sync m16n8k32 and m16n8k16 with s8 operands.
//
// A block: WR = 64 weight rows of one matrix x up to 16 NJ of its activation rows (blockIdx.y walks a matrix's rows
// in such chunks).  K goes in stages of 256 values; a type of 32-value blocks may end inside the last one
// (gemm_iq_i8_takes()).  Per stage every thread decodes WR / 32 sub-blocks of one weight
// row into shared memory, the stage's activations are copied beside them, and a warp multiplies its WR / 4 weight
// rows with its tiles of 8 activation rows: one MMA per 32 values, or per 16 for a type with a scale per 16 values.
// A tile takes most of the shared memory a multiprocessor has, so a block waits alone for what it reads from the
// card's memory: a thread issues all of a stage's reads together, into registers, and the next stage's before it
// multiplies this one, so the wait is behind the products.  (Read and stored one piece after the other, the reads
// were more than half of a launch's time: the experts of Qwen3.6 UD-Q3_K_M, 256 of 64 rows each, took 20-30% longer.)
// An MMA's int32 sums become floats at once, sum * (the weight sub-block's scale) * (the activation block's scale),
// and are added to the thread's FP32 sums; a type with a minimum per sub-block adds min * (the activation block's
// scale times the sum of its codes).  A value's sum is its own, stage after stage, so it does not depend on the
// launch its row is in.
//
// The activations (gemm_iq_i8_quantize): [rows][K] int8 codes, then [rows][K / 32] float scales, then [rows][K / 32]
// floats scale * (sum of the block's codes).  A block of 32 values: scale = max|x| / 127, code = round(x / scale).
constexpr int I8M_NT = 256;
constexpr int I8M_LD = 68;   // words of a shared row of codes: 64 + 4, so the 8 rows a warp reads at once lie in 8 banks

#define STRATA_I8M_TYPES(X) X(2) X(6) X(7) X(8) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42)
__host__ __device__ constexpr bool i8m_type(int t) { return t == 11 || i8_block_bytes(t) != 0; }
__host__ __device__ constexpr bool i8m_two(int t) { return t == 11 || i8_two_scales(t); }   // a scale per 16 values

template <int T, int NJ, int WR>
struct I8mSmem {
    static constexpr int MT = 16 * NJ;                                           // activation rows of a tile
    static constexpr int GW = !i8_gg_type(T) && i8_grid_words(T) > 4 ? (i8_grid_words(T) + 3) / 4 * 4 : 0;
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int NS = i8m_two(T) ? 16 : 8;                               // scales of a weight row a stage
    static constexpr int LDS = NS + 1, LDB = 9;                                  // odd strides: no two of a warp's reads in one bank
    static constexpr int MN = i8_min_type(T) ? 1 : 0;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_x = o_w + WR * I8M_LD, o_ws = o_x + MT * I8M_LD, o_wm = o_ws + WR * LDS,
                         o_xd = o_wm + MN * WR * LDB, o_xs = o_xd + MT * LDB, o_id = o_xs + MN * MT * LDB,
                         total = (o_id + MT + 3) / 4 * 4;
    static constexpr size_t bytes = (size_t) total * 4;
};

__device__ __forceinline__ void i8m_mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void i8m_mma16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// Q3_K (block_q3_K, 110 bytes: hmask 32, qs 64, 12 scale bytes, d), sub-block ib (0..7) of super-block sb of the row:
// value l (0..31) = (the 2 bits of qs[32 (ib >> 2) + l] at 2 (ib & 3)) + 4 (bit ib of hmask[l]) - 4, times d and the
// 6-bit scale of its half less 32.  Scale 4 a + c has its low 4 bits in nibble (a >> 1) of scales[4 (a & 1) + c] and
// its high 2 bits at 2 a of scales[8 + c] (dq_q3_k above reads the same bits).  The words as they are read: 8 of
// qs, 8 of hmask, the 12 scale bytes, d.
constexpr int I8M_Q3K_RAW = 20;
__device__ __forceinline__ void i8m_q3k_load(I8P row, int sb, int ib, uint32_t (&r)[I8M_Q3K_RAW]) {
    I8P b = row + (size_t) sb * 110;
    I8P qs = b + 32 + 32 * (ib >> 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        r[k] = i8_ld32(qs + 4 * k);
        r[8 + k] = i8_ld32(b + 4 * k);
    }
    r[16] = i8_ld32(b + 96);
    r[17] = i8_ld32(b + 100);
    r[18] = i8_ld32(b + 104);
    r[19] = i8_ld16(b + 108);
}
__device__ __forceinline__ void i8m_q3k_convert(const uint32_t (&r)[I8M_Q3K_RAW], int ib, uint32_t (&q)[8], float& f0, float& f1) {
    const int sh = 2 * (ib & 3);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t lo = (r[k] >> sh) & 0x03030303u, hi = ((r[8 + k] >> ib) & 0x01010101u) << 2;
        q[k] = (((lo | hi) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;   // 0..7 less 4 in each byte, no borrow between bytes
    }
    const int a = ib >> 1, c = 2 * (ib & 1);
    const uint32_t lw = ((a & 1) ? r[17] : r[16]) >> (4 * (a >> 1));
    const int u0 = (int) (((lw >> (8 * c)) & 15u) | (((r[18] >> (8 * c + 2 * a)) & 3u) << 4));
    const int u1 = (int) (((lw >> (8 * c + 8)) & 15u) | (((r[18] >> (8 * c + 8 + 2 * a)) & 3u) << 4));
    const float d = i8_half(r[19]);
    f0 = d * (float) (u0 - 32);
    f1 = d * (float) (u1 - 32);
}
template <int T> constexpr int I8M_RAW = T == 11 ? I8M_Q3K_RAW : I8NR<T>;   // words a thread reads for a sub-block

// One block a row, a warp a block of 32 values at a time.
__global__ void __launch_bounds__(I8M_NT) i8m_quantize_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, long long ld,
                                                             int K, long long rows, uint32_t* __restrict__ q, float* __restrict__ d,
                                                             float* __restrict__ ds) {
    const long long row = (long long) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) return;
    const float* src = x + (ids ? (long long) ids[row] : row) * ld;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = K / 32;
    for (int b = warp; b < nb; b += I8M_NT / 32) {
        const float v = src[b * 32 + lane];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d_inv = 127.0f / amax;
        const int code = amax == 0.0f ? 0 : (int) roundf(v * d_inv);
        int sum = code;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        uint32_t w = (uint32_t) (code & 255) << (8 * (lane & 3));
        w |= __shfl_xor_sync(0xffffffffu, w, 1);
        w |= __shfl_xor_sync(0xffffffffu, w, 2);
        if ((lane & 3) == 0) q[(row * K + b * 32) / 4 + (lane >> 2)] = w;
        if (lane == 0) {
            const float scale = amax == 0.0f ? 0.0f : 1.0f / d_inv;
            d[row * nb + b] = scale;
            ds[row * nb + b] = scale * (float) sum;
        }
    }
}

template <int T, int NJ, int WR>
__global__ void __launch_bounds__(I8M_NT) i8m_kernel(const uint8_t* __restrict__ W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                                                    const uint8_t* __restrict__ X, const float* __restrict__ XD,
                                                    const float* __restrict__ XS, const int32_t* __restrict__ bounds, int ch_per_e,
                                                    const int32_t* __restrict__ ids, const int32_t* __restrict__ from,
                                                    float* __restrict__ Y, long long ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800   // (a pass for an older card compiles this empty: the host asks i8m_arch_kernel first)
    using SM = I8mSmem<T, NJ, WR>;
    constexpr int MT = SM::MT, LDS = SM::LDS, LDB = SM::LDB;
    constexpr int RT = I8M_NT / WR;   // threads of a weight row: each decodes 8 / RT of its 8 sub-blocks a stage
    constexpr int NA = WR / 64;       // tiles of 16 weight rows a warp multiplies
    constexpr bool TWO = i8m_two(T), HAS_MIN = i8_min_type(T);
    extern __shared__ uint4 i8m_dyn[];
    uint32_t* const smem = (uint32_t*) i8m_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    uint32_t* const s_w = smem + SM::o_w;                    // [WR][I8M_LD]: a stage of the weight rows' codes
    uint32_t* const s_x = smem + SM::o_x;                    // [MT][I8M_LD]: a stage of the activation rows' codes
    float* const s_ws = (float*) (smem + SM::o_ws);          // [WR][LDS]: scale of (weight row, sub-block or half of one)
    float* const s_wm = (float*) (smem + SM::o_wm);          // [WR][LDB]: minimum of (weight row, sub-block)
    float* const s_xd = (float*) (smem + SM::o_xd);          // [MT][LDB]: scale of (activation row, block)
    float* const s_xs = (float*) (smem + SM::o_xs);          // [MT][LDB]: scale * sum of codes
    int* const s_id = (int*) (smem + SM::o_id);              // [MT]: the row of Y, -1 past the last

    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int ntok = min(MT, r1 - c0);
    const int n0 = blockIdx.x * WR;
    const int tid = threadIdx.x;

    const uint32_t* grid_rd = i8_grid_src<T>();
    if constexpr (SM::GW > 0) {
        for (int i = tid; i < i8_grid_words(T); i += I8M_NT) s_grid[i] = grid_rd[i];
        grid_rd = s_grid;
    }
    if constexpr (i8_sign_type(T)) {   // entry tid: 8 sign bits (the ksigns types: 7 bits and their parity) as byte masks, i8_kernel's table
        const uint32_t b = (uint32_t) tid;
        uint32_t sb = b;
        if constexpr (T != 21 && T != 22) sb = (b & 127u) | (((uint32_t) __popc(b & 127u) & 1u) << 7);
        uint32_t mlo = 0, mhi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            mlo |= ((sb >> k) & 1u) ? (0xFFu << (8 * k)) : 0u;
            mhi |= ((sb >> (4 + k)) & 1u) ? (0xFFu << (8 * k)) : 0u;
        }
        *(uint2*) (s_sg + 2 * b) = make_uint2(mlo, mhi);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (T == 20 || T == 23) {
        const uint32_t* t = (const uint32_t*) kvalues_iq4nl;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv[i] = t[i];
    } else if constexpr (T == 2) {   // Q4_0: a nibble less 8, IQ4_NL's blocks otherwise
        kv[0] = 0xFBFAF9F8u; kv[1] = 0xFFFEFDFCu; kv[2] = 0x03020100u; kv[3] = 0x07060504u;
    }
    if (tid < MT) s_id[tid] = tid < ntok ? (ids ? ids[c0 + tid] : c0 + tid) : -1;

    // this thread's weight sub-blocks: row wr of the tile, its sub-blocks r0, r0 + RT, ... of a stage's 8
    const int wr = tid / RT, r0 = tid % RT;
    const bool w_in = n0 + wr < n_out;
    I8P const blob = I8_GPTR(W + (size_t) e * mat_bytes + (size_t) n0 * row_bytes);
    const uint32_t row_off = (uint32_t) ((size_t) wr * row_bytes);
    // its tensor-core tiles: weight rows i0 + 16 a + g and 8 more, activation rows j0 + g as columns, j0 = 8 (2 n + jp)
    const int warp = tid >> 5, g = (tid & 31) >> 2, tg = tid & 3;
    const int i0 = (warp & 3) * 16 * NA, jp = warp >> 2;
    const uint32_t* const wa = s_w + (i0 + g) * I8M_LD + tg;
    const float* const sa = s_ws + (i0 + g) * LDS;
    const float* const sm = s_wm + (i0 + g) * LDB;

    float acc[NA][NJ][4];
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
        for (int n = 0; n < NJ; ++n)
#pragma unroll
            for (int l = 0; l < 4; ++l) acc[a][n][l] = 0.0f;

    // A stage as it is read: this thread's sub-blocks of its weight row, its 16-byte pieces of the activation rows
    // and its activation scales.
    constexpr int PER = 8 / RT, NSC = (MT * 8 + I8M_NT - 1) / I8M_NT;
    uint32_t wraw[PER][I8M_RAW<T>];
    uint4 xr[NJ];
    float xdr[NSC], xsr[NSC];
    // the rows of X this thread reads: the launch's row, or the one `from` gives for it
    int xrow[NJ], srow[NSC];
#pragma unroll
    for (int n = 0; n < NJ; ++n) {
        const int tok = 16 * n + (tid >> 4);
        xrow[n] = tok < ntok ? (from ? from[c0 + tok] : c0 + tok) : 0;
    }
#pragma unroll
    for (int c = 0; c < NSC; ++c) {
        const int i = tid + c * I8M_NT, tok = i >> 3;
        srow[c] = i < MT * 8 && tok < ntok ? (from ? from[c0 + tok] : c0 + tok) : 0;
    }
    // A row of 32-value blocks may end inside its last stage (Gemma 4 26B's down: 704 values): the sub-blocks past
    // its end are not read, and go in as zeros, codes and scales, so they add nothing.
    const auto here = [&](int S, int ib) { return S * 256 + ib * 32 < K; };
    const auto fetch = [&](int S) {
        if (w_in) {
#pragma unroll
            for (int part = 0; part < PER; ++part) {
                const int ib = r0 + RT * part;
                if (!here(S, ib)) continue;
                if constexpr (T == 11) i8m_q3k_load(blob + row_off, S, ib, wraw[part]);
                else i8_load<T>(blob, i8_offs<T>(row_off, ib & 3), 2 * S + (ib >> 2), wraw[part]);
            }
        }
#pragma unroll
        for (int n = 0; n < NJ; ++n) {   // 16 bytes of activation row 16 n + tid / 16
            const int tok = 16 * n + (tid >> 4);
            xr[n] = make_uint4(0, 0, 0, 0);
            if (tok < ntok && S * 256 + (tid & 15) * 16 < K)
                xr[n] = *(const uint4*) (X + (size_t) xrow[n] * K + (size_t) S * 256 + (tid & 15) * 16);
        }
#pragma unroll
        for (int c = 0; c < NSC; ++c) {
            const int i = tid + c * I8M_NT, tok = i >> 3;
            const size_t at = (size_t) srow[c] * (K / 32) + (size_t) S * 8 + (i & 7);
            xdr[c] = xsr[c] = 0.0f;
            if (i < MT * 8 && tok < ntok && here(S, i & 7)) {
                xdr[c] = XD[at];
                if constexpr (HAS_MIN) xsr[c] = XS[at];
            }
        }
    };
    const auto put = [&](int S) {
#pragma unroll
        for (int part = 0; part < PER; ++part) {
            const int ib = r0 + RT * part;
            uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float f0 = 0.0f, f1 = 0.0f, mn = 0.0f;
            if (w_in && here(S, ib)) {
                if constexpr (T == 11) {
                    i8m_q3k_convert(wraw[part], ib, q, f0, f1);
                } else {
                    uint32_t w[I8NW<T>];
                    int m0, m1;
                    float base;
                    i8_pack<T>(wraw[part], ib, w);
                    i8_convert<T>(w, grid_rd, (const uint2*) s_sg, kv, q, m0, m1, base, ib, &mn);
                    f0 = base * (float) m0;
                    f1 = base * (float) m1;
                }
            }
            uint4* dst = (uint4*) (s_w + wr * I8M_LD + ib * 8);
            dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
            dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
            if constexpr (TWO) {
                s_ws[wr * LDS + 2 * ib] = f0;
                s_ws[wr * LDS + 2 * ib + 1] = f1;
            } else {
                s_ws[wr * LDS + ib] = f0;
            }
            if constexpr (HAS_MIN) s_wm[wr * LDB + ib] = mn;
        }
#pragma unroll
        for (int n = 0; n < NJ; ++n) *(uint4*) (s_x + (16 * n + (tid >> 4)) * I8M_LD + (tid & 15) * 4) = xr[n];
#pragma unroll
        for (int c = 0; c < NSC; ++c) {
            const int i = tid + c * I8M_NT;
            if (i < MT * 8) {
                s_xd[(i >> 3) * LDB + (i & 7)] = xdr[c];
                if constexpr (HAS_MIN) s_xs[(i >> 3) * LDB + (i & 7)] = xsr[c];
            }
        }
    };

    const int stages = (K + 255) / 256;
    fetch(0);
    for (int S = 0; S < stages; ++S) {
        __syncthreads();   // the stage before is multiplied (and the tables are written)
        put(S);
        __syncthreads();
        if (S + 1 < stages) fetch(S + 1);   // read while this stage is multiplied

        if constexpr (TWO) {
#pragma unroll
            for (int kq = 0; kq < 16; ++kq) {
                uint32_t a0[NA], a1[NA];
                float w0[NA], w1[NA];
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    a0[a] = wa[16 * a * I8M_LD + kq * 4];
                    a1[a] = wa[(16 * a + 8) * I8M_LD + kq * 4];
                    w0[a] = sa[16 * a * LDS + kq];
                    w1[a] = sa[(16 * a + 8) * LDS + kq];
                }
#pragma unroll
                for (int n = 0; n < NJ; ++n) {
                    const int j0 = 8 * (2 * n + jp);
                    if (j0 >= ntok) break;
                    const uint32_t b0 = s_x[(j0 + g) * I8M_LD + kq * 4 + tg];
                    const int xa = (j0 + 2 * tg) * LDB + (kq >> 1);
                    const float x0 = s_xd[xa], x1 = s_xd[xa + LDB];
#pragma unroll
                    for (int a = 0; a < NA; ++a) {
                        int c[4] = {0, 0, 0, 0};
                        i8m_mma16(c, a0[a], a1[a], b0);
                        acc[a][n][0] = fmaf((float) c[0], w0[a] * x0, acc[a][n][0]);
                        acc[a][n][1] = fmaf((float) c[1], w0[a] * x1, acc[a][n][1]);
                        acc[a][n][2] = fmaf((float) c[2], w1[a] * x0, acc[a][n][2]);
                        acc[a][n][3] = fmaf((float) c[3], w1[a] * x1, acc[a][n][3]);
                    }
                }
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                uint32_t a0[NA], a1[NA], a2[NA], a3[NA];
                float w0[NA], w1[NA], m0[NA], m1[NA];
#pragma unroll
                for (int a = 0; a < NA; ++a) {
                    a0[a] = wa[16 * a * I8M_LD + kb * 8];
                    a1[a] = wa[(16 * a + 8) * I8M_LD + kb * 8];
                    a2[a] = wa[16 * a * I8M_LD + kb * 8 + 4];
                    a3[a] = wa[(16 * a + 8) * I8M_LD + kb * 8 + 4];
                    w0[a] = sa[16 * a * LDS + kb];
                    w1[a] = sa[(16 * a + 8) * LDS + kb];
                    m0[a] = m1[a] = 0.0f;
                    if constexpr (HAS_MIN) {
                        m0[a] = sm[16 * a * LDB + kb];
                        m1[a] = sm[(16 * a + 8) * LDB + kb];
                    }
                }
#pragma unroll
                for (int n = 0; n < NJ; ++n) {
                    const int j0 = 8 * (2 * n + jp);
                    if (j0 >= ntok) break;
                    const uint32_t* xb = s_x + (j0 + g) * I8M_LD + kb * 8 + tg;
                    const uint32_t b0 = xb[0], b1 = xb[4];
                    const int xa = (j0 + 2 * tg) * LDB + kb;
                    const float x0 = s_xd[xa], x1 = s_xd[xa + LDB];
                    float s0 = 0.0f, s1 = 0.0f;
                    if constexpr (HAS_MIN) {
                        s0 = s_xs[xa];
                        s1 = s_xs[xa + LDB];
                    }
#pragma unroll
                    for (int a = 0; a < NA; ++a) {
                        int c[4] = {0, 0, 0, 0};
                        i8m_mma32(c, a0[a], a1[a], a2[a], a3[a], b0, b1);
                        acc[a][n][0] = fmaf((float) c[0], w0[a] * x0, acc[a][n][0]);
                        acc[a][n][1] = fmaf((float) c[1], w0[a] * x1, acc[a][n][1]);
                        acc[a][n][2] = fmaf((float) c[2], w1[a] * x0, acc[a][n][2]);
                        acc[a][n][3] = fmaf((float) c[3], w1[a] * x1, acc[a][n][3]);
                        if constexpr (HAS_MIN) {
                            acc[a][n][0] = fmaf(m0[a], s0, acc[a][n][0]);
                            acc[a][n][1] = fmaf(m0[a], s1, acc[a][n][1]);
                            acc[a][n][2] = fmaf(m1[a], s0, acc[a][n][2]);
                            acc[a][n][3] = fmaf(m1[a], s1, acc[a][n][3]);
                        }
                    }
                }
            }
        }
    }

    // a thread holds weight rows i0 + 16 a + g and 8 more for activation rows j0 + 2 tg and j0 + 2 tg + 1
#pragma unroll
    for (int n = 0; n < NJ; ++n) {
        const int j0 = 8 * (2 * n + jp);
        if (j0 >= ntok) break;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int id = s_id[j0 + 2 * tg + l];
            if (id < 0) continue;
            float* out = Y + (size_t) id * (size_t) ldy;
#pragma unroll
            for (int a = 0; a < NA; ++a) {
                const int ia = n0 + i0 + 16 * a + g;
                if (ia < n_out) out[ia] = acc[a][n][l];
                if (ia + 8 < n_out) out[ia + 8] = acc[a][n][2 + l];
            }
        }
    }
#endif
}

// the architecture the code running on this card was compiled for (a build may hold no pass for the card's own)
__global__ void i8m_arch_kernel(int* out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__;
#else
    *out = 0;
#endif
}

struct I8mCard {
    bool read = false, on = false;
    size_t shared = 0;
};

// the current card: int8 tensor cores in the code it runs, and the shared memory it gives a block
I8mCard i8m_card() {
    static I8mCard known[64];
    static std::mutex mu;
    static const bool off = [] { const char* v = std::getenv("STRATA_PF_I8MMA"); return v != nullptr && v[0] == '0'; }();
    int dev = 0;
    if (off || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return I8mCard{};
    std::lock_guard<std::mutex> lock(mu);
    if (!known[dev].read) {
        I8mCard c;
        c.read = true;
        int v = 0, *d_arch = nullptr, arch = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess) c.shared = (size_t) v;
        if (cudaMalloc(&d_arch, sizeof(int)) == cudaSuccess) {
            i8m_arch_kernel<<<1, 1>>>(d_arch);
            if (cudaMemcpy(&arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) arch = 0;
            cudaFree(d_arch);
        }
        (void) cudaGetLastError();
        c.on = arch >= 800;
        known[dev] = c;
    }
    return known[dev];
}

template <int T, int NJ, int WR>
bool i8m_launch(int dev, dim3 grid, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int ch, const int32_t* ids,
                const int32_t* from, float* Y, long long ldy) {
    constexpr size_t bytes = I8mSmem<T, NJ, WR>::bytes;
    static bool attr[64];   // the attribute belongs to a card (see gemm_iq_f16_grouped)
    if (!attr[dev]) {
        cudaFuncSetAttribute(i8m_kernel<T, NJ, WR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
        attr[dev] = true;
    }
    i8m_kernel<T, NJ, WR><<<grid, I8M_NT, bytes, s>>>(W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, ch, ids, from, Y,
                                                     ldy);
    return cudaGetLastError() == cudaSuccess;
}

// 64 weight rows a block: 128 measured slower on an RTX PRO 6000 (a dense Q8_0 [2048 -> 8192] times 2,048 rows 0.61 ms
// against 0.46, the experts no faster), so only 64 is built
#define STRATA_I8M_TILES(X) X(1, 64) X(2, 64) X(4, 64) X(8, 64)

template <int T>
size_t i8m_tile_bytes(int nj, int wr) {
#define STRATA_I8M_SIZE(NJ_, WR_) if (nj == NJ_ && wr == WR_) return I8mSmem<T, NJ_, WR_>::bytes;
    STRATA_I8M_TILES(STRATA_I8M_SIZE)
#undef STRATA_I8M_SIZE
    return (size_t) -1;
}

// the tile as wide in activation rows as the launch's rows need, of what the card has the shared memory for
template <int T>
bool i8m_launch_type(const I8mCard& card, int dev, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                     const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int n_mats, int max_rows,
                     const int32_t* ids, const int32_t* from, float* Y, long long ldy) {
    int nj = max_rows <= 16 ? 1 : max_rows <= 32 ? 2 : max_rows <= 64 ? 4 : 8;
    const int wr = 64;
    while (nj > 1 && i8m_tile_bytes<T>(nj, wr) > card.shared) nj /= 2;
    if (i8m_tile_bytes<T>(nj, wr) > card.shared) return false;
    const long long ch = (max_rows + 16 * nj - 1) / (16 * nj), gy = (long long) n_mats * ch;
    if (gy > 65535) return false;
    const dim3 grid((unsigned) ((n_out + wr - 1) / wr), (unsigned) gy);
#define STRATA_I8M_GO(NJ_, WR_) \
    if (nj == NJ_ && wr == WR_) \
        return i8m_launch<T, NJ_, WR_>(dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, ids, from, \
                                       Y, ldy);
    STRATA_I8M_TILES(STRATA_I8M_GO)
#undef STRATA_I8M_GO
    return false;
}
