// src/kernels/cuda/i8_mma.cuh - a group of quantized matrices times rows of int8 activations on the int8 tensor cores
// (gemm_iq_i8_grouped, declared in include/strata/kernels/iq_kernels.hpp), for a matrix of any size.  Included once by
// iq_kernels.cu inside its anonymous namespace, after i8_expert.cuh: a weight sub-block is decoded by that file's
// i8_load / i8_pack / i8_convert (32 values as 8 words of int8, their integer scales and float base), Q3_K by the
// decoder here.  CUDA only, Ampere and newer: mma.sync m16n8k32 and m16n8k16 with s8 operands.
//
// A block: 64 weight rows of one matrix x up to 16 NJ of its activation rows (blockIdx.y walks a matrix's rows in such
// chunks).  K goes in stages of 256 values.  Per stage every thread decodes two sub-blocks of one weight row into shared
// memory, the stage's activations are copied beside them, and a warp multiplies its 16 weight rows with its tiles of 8
// activation rows: one MMA per 32 values, or per 16 for a type with a scale per 16 values.  An MMA's int32 sums become
// floats at once, sum * (the weight sub-block's scale) * (the activation block's scale), and are added to the thread's
// FP32 sums; a type with a minimum per sub-block adds min * (the activation block's scale times the sum of its codes).
// A value's sum is its own, stage after stage, so it does not depend on the launch its row is in.
//
// The activations (gemm_iq_i8_quantize): [rows][K] int8 codes, then [rows][K / 32] float scales, then [rows][K / 32]
// floats scale * (sum of the block's codes).  A block of 32 values: scale = max|x| / 127, code = round(x / scale).
constexpr int I8M_NT = 256;
constexpr int I8M_LD = 68;   // words of a shared row of codes: 64 + 4, so the 8 rows a warp reads at once lie in 8 banks

#define STRATA_I8M_TYPES(X) X(6) X(7) X(8) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42)
__host__ __device__ constexpr bool i8m_type(int t) { return t == 11 || i8_block_bytes(t) != 0; }
__host__ __device__ constexpr bool i8m_two(int t) { return t == 11 || i8_two_scales(t); }   // a scale per 16 values

template <int T, int NJ>
struct I8mSmem {
    static constexpr int MT = 16 * NJ;                                           // activation rows of a tile
    static constexpr int GW = !i8_gg_type(T) && i8_grid_words(T) > 4 ? (i8_grid_words(T) + 3) / 4 * 4 : 0;
    static constexpr int SG = i8_sign_type(T) ? 512 : 0;
    static constexpr int NS = i8m_two(T) ? 16 : 8;                               // scales of a weight row a stage
    static constexpr int LDS = NS + 1, LDB = 9;                                  // odd strides: no two of a warp's reads in one bank
    static constexpr int MN = i8_min_type(T) ? 1 : 0;
    static constexpr int o_sg = GW, o_w = o_sg + SG, o_x = o_w + 64 * I8M_LD, o_ws = o_x + MT * I8M_LD, o_wm = o_ws + 64 * LDS,
                         o_xd = o_wm + MN * 64 * LDB, o_xs = o_xd + MT * LDB, o_id = o_xs + MN * MT * LDB,
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
// its high 2 bits at 2 a of scales[8 + c] (dq_q3_k above reads the same bits).
__device__ __forceinline__ void i8m_q3k(I8P row, int sb, int ib, uint32_t (&q)[8], float& f0, float& f1) {
    I8P b = row + (size_t) sb * 110;
    I8P qs = b + 32 + 32 * (ib >> 2);
    const int sh = 2 * (ib & 3);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t lo = (i8_ld32(qs + 4 * k) >> sh) & 0x03030303u, hi = ((i8_ld32(b + 4 * k) >> ib) & 0x01010101u) << 2;
        q[k] = (((lo | hi) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;   // 0..7 less 4 in each byte, no borrow between bytes
    }
    const uint32_t s2 = i8_ld32(b + 104);
    const int a = ib >> 1, c = 2 * (ib & 1);
    const uint32_t lw = i8_ld32(b + 96 + 4 * (a & 1)) >> (4 * (a >> 1));
    const int u0 = (int) (((lw >> (8 * c)) & 15u) | (((s2 >> (8 * c + 2 * a)) & 3u) << 4));
    const int u1 = (int) (((lw >> (8 * c + 8)) & 15u) | (((s2 >> (8 * c + 8 + 2 * a)) & 3u) << 4));
    const float d = i8_half(i8_ld16(b + 108));
    f0 = d * (float) (u0 - 32);
    f1 = d * (float) (u1 - 32);
}

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

template <int T, int NJ>
__global__ void __launch_bounds__(I8M_NT) i8m_kernel(const uint8_t* __restrict__ W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                                                    const uint8_t* __restrict__ X, const float* __restrict__ XD,
                                                    const float* __restrict__ XS, const int32_t* __restrict__ bounds, int ch_per_e,
                                                    const int32_t* __restrict__ ids, float* __restrict__ Y, long long ldy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800   // (a pass for an older card compiles this empty: the host asks i8m_arch_kernel first)
    using SM = I8mSmem<T, NJ>;
    constexpr int MT = SM::MT, LDS = SM::LDS, LDB = SM::LDB;
    constexpr bool TWO = i8m_two(T), HAS_MIN = i8_min_type(T);
    extern __shared__ uint4 i8m_dyn[];
    uint32_t* const smem = (uint32_t*) i8m_dyn;
    uint32_t* const s_grid = smem;
    uint32_t* const s_sg = smem + SM::o_sg;
    uint32_t* const s_w = smem + SM::o_w;                    // [64][I8M_LD]: a stage of the weight rows' codes
    uint32_t* const s_x = smem + SM::o_x;                    // [MT][I8M_LD]: a stage of the activation rows' codes
    float* const s_ws = (float*) (smem + SM::o_ws);          // [64][LDS]: scale of (weight row, sub-block or half of one)
    float* const s_wm = (float*) (smem + SM::o_wm);          // [64][LDB]: minimum of (weight row, sub-block)
    float* const s_xd = (float*) (smem + SM::o_xd);          // [MT][LDB]: scale of (activation row, block)
    float* const s_xs = (float*) (smem + SM::o_xs);          // [MT][LDB]: scale * sum of codes
    int* const s_id = (int*) (smem + SM::o_id);              // [MT]: the row of Y, -1 past the last

    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int ntok = min(MT, r1 - c0);
    const int n0 = blockIdx.x * 64;
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
    }
    if (tid < MT) s_id[tid] = tid < ntok ? (ids ? ids[c0 + tid] : c0 + tid) : -1;

    // this thread's weight sub-blocks: row wr of the tile, sub-block wp of each half of a stage
    const int wr = tid >> 2, wp = tid & 3;
    const bool w_in = n0 + wr < n_out;
    I8P const blob = I8_GPTR(W + (size_t) e * mat_bytes + (size_t) n0 * row_bytes);
    const uint32_t row_off = (uint32_t) ((size_t) wr * row_bytes);
    // its tensor-core tiles: weight rows i0 + g and i0 + g + 8, activation rows j0 + g as columns, j0 = 8 (2 n + jp)
    const int warp = tid >> 5, g = (tid & 31) >> 2, tg = tid & 3;
    const int i0 = (warp & 3) * 16, jp = warp >> 2;
    const uint32_t* const wa = s_w + (i0 + g) * I8M_LD + tg;
    const float* const sa = s_ws + (i0 + g) * LDS;
    const float* const sm = s_wm + (i0 + g) * LDB;

    float acc[NJ][4];
#pragma unroll
    for (int n = 0; n < NJ; ++n)
#pragma unroll
        for (int l = 0; l < 4; ++l) acc[n][l] = 0.0f;

    for (int S = 0; S < K / 256; ++S) {
        __syncthreads();   // the stage before is multiplied (and the tables are written)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int ib = 4 * h + wp;
            uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float f0 = 0.0f, f1 = 0.0f, mn = 0.0f;
            if (w_in) {
                if constexpr (T == 11) {
                    i8m_q3k(blob + row_off, S, ib, q, f0, f1);
                } else {
                    uint32_t r[I8NR<T>], w[I8NW<T>];
                    int m0, m1;
                    float base;
                    i8_load<T>(blob, i8_offs<T>(row_off, wp), 2 * S + h, r);
                    i8_pack<T>(r, ib, w);
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
        for (int n = 0; n < NJ; ++n) {   // 16 bytes of activation row 16 n + tid / 16
            const int tok = 16 * n + (tid >> 4), part = tid & 15;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (tok < ntok) v = *(const uint4*) (X + (size_t) (c0 + tok) * K + (size_t) S * 256 + part * 16);
            *(uint4*) (s_x + tok * I8M_LD + part * 4) = v;
        }
        for (int i = tid; i < MT * 8; i += I8M_NT) {
            const int tok = i >> 3, b = i & 7;
            const size_t at = (size_t) (c0 + tok) * (K / 32) + (size_t) S * 8 + b;
            s_xd[tok * LDB + b] = tok < ntok ? XD[at] : 0.0f;
            if constexpr (HAS_MIN) s_xs[tok * LDB + b] = tok < ntok ? XS[at] : 0.0f;
        }
        __syncthreads();

        if constexpr (TWO) {
#pragma unroll
            for (int kq = 0; kq < 16; ++kq) {
                const uint32_t a0 = wa[kq * 4], a1 = wa[8 * I8M_LD + kq * 4];
                const float w0 = sa[kq], w1 = sa[8 * LDS + kq];
#pragma unroll
                for (int n = 0; n < NJ; ++n) {
                    const int j0 = 8 * (2 * n + jp);
                    if (j0 >= ntok) break;
                    int c[4] = {0, 0, 0, 0};
                    i8m_mma16(c, a0, a1, s_x[(j0 + g) * I8M_LD + kq * 4 + tg]);
                    const float x0 = s_xd[(j0 + 2 * tg) * LDB + (kq >> 1)], x1 = s_xd[(j0 + 2 * tg + 1) * LDB + (kq >> 1)];
                    acc[n][0] = fmaf((float) c[0], w0 * x0, acc[n][0]);
                    acc[n][1] = fmaf((float) c[1], w0 * x1, acc[n][1]);
                    acc[n][2] = fmaf((float) c[2], w1 * x0, acc[n][2]);
                    acc[n][3] = fmaf((float) c[3], w1 * x1, acc[n][3]);
                }
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                const uint32_t a0 = wa[kb * 8], a1 = wa[8 * I8M_LD + kb * 8], a2 = wa[kb * 8 + 4], a3 = wa[8 * I8M_LD + kb * 8 + 4];
                const float w0 = sa[kb], w1 = sa[8 * LDS + kb];
                float m0 = 0.0f, m1 = 0.0f;
                if constexpr (HAS_MIN) {
                    m0 = sm[kb];
                    m1 = sm[8 * LDB + kb];
                }
#pragma unroll
                for (int n = 0; n < NJ; ++n) {
                    const int j0 = 8 * (2 * n + jp);
                    if (j0 >= ntok) break;
                    const uint32_t* xb = s_x + (j0 + g) * I8M_LD + kb * 8 + tg;
                    int c[4] = {0, 0, 0, 0};
                    i8m_mma32(c, a0, a1, a2, a3, xb[0], xb[4]);
                    const int xa = (j0 + 2 * tg) * LDB + kb;
                    const float x0 = s_xd[xa], x1 = s_xd[xa + LDB];
                    acc[n][0] = fmaf((float) c[0], w0 * x0, acc[n][0]);
                    acc[n][1] = fmaf((float) c[1], w0 * x1, acc[n][1]);
                    acc[n][2] = fmaf((float) c[2], w1 * x0, acc[n][2]);
                    acc[n][3] = fmaf((float) c[3], w1 * x1, acc[n][3]);
                    if constexpr (HAS_MIN) {
                        const float s0 = s_xs[xa], s1 = s_xs[xa + LDB];
                        acc[n][0] = fmaf(m0, s0, acc[n][0]);
                        acc[n][1] = fmaf(m0, s1, acc[n][1]);
                        acc[n][2] = fmaf(m1, s0, acc[n][2]);
                        acc[n][3] = fmaf(m1, s1, acc[n][3]);
                    }
                }
            }
        }
    }

    // a thread holds weight rows i0 + g and i0 + g + 8 for activation rows j0 + 2 tg and j0 + 2 tg + 1
    const int ia = n0 + i0 + g, ib = ia + 8;
#pragma unroll
    for (int n = 0; n < NJ; ++n) {
        const int j0 = 8 * (2 * n + jp);
        if (j0 >= ntok) break;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int id = s_id[j0 + 2 * tg + l];
            if (id < 0) continue;
            float* out = Y + (size_t) id * (size_t) ldy;
            if (ia < n_out) out[ia] = acc[n][l];
            if (ib < n_out) out[ib] = acc[n][2 + l];
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

template <int T, int NJ>
bool i8m_launch(const I8mCard& card, int dev, dim3 grid, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int ch, const int32_t* ids, float* Y,
                long long ldy) {
    constexpr size_t bytes = I8mSmem<T, NJ>::bytes;
    if (bytes > card.shared) return false;
    static bool attr[64];   // the attribute belongs to a card (see gemm_iq_f16_grouped)
    if (!attr[dev]) {
        cudaFuncSetAttribute(i8m_kernel<T, NJ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
        attr[dev] = true;
    }
    i8m_kernel<T, NJ><<<grid, I8M_NT, bytes, s>>>(W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, ch, ids, Y, ldy);
    return cudaGetLastError() == cudaSuccess;
}

// the widest tile the launch's rows need that the card has the shared memory for
template <int T>
bool i8m_launch_type(const I8mCard& card, int dev, cudaStream_t s, const uint8_t* W, size_t mat_bytes, size_t row_bytes, int n_out, int K,
                     const uint8_t* X, const float* XD, const float* XS, const int32_t* bounds, int n_mats, int max_rows,
                     const int32_t* ids, float* Y, long long ldy) {
    int nj = max_rows <= 16 ? 1 : max_rows <= 32 ? 2 : max_rows <= 64 ? 4 : 8;
    while (nj > 1 && (nj == 8 ? I8mSmem<T, 8>::bytes : nj == 4 ? I8mSmem<T, 4>::bytes : I8mSmem<T, 2>::bytes) > card.shared) nj /= 2;
    const long long ch = (max_rows + 16 * nj - 1) / (16 * nj), gy = (long long) n_mats * ch;
    if (gy > 65535) return false;
    const dim3 grid((unsigned) ((n_out + 63) / 64), (unsigned) gy);
    switch (nj) {
        case 1: return i8m_launch<T, 1>(card, dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, ids, Y, ldy);
        case 2: return i8m_launch<T, 2>(card, dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, ids, Y, ldy);
        case 4: return i8m_launch<T, 4>(card, dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, ids, Y, ldy);
        default: return i8m_launch<T, 8>(card, dev, grid, s, W, mat_bytes, row_bytes, n_out, K, X, XD, XS, bounds, (int) ch, ids, Y, ldy);
    }
}
