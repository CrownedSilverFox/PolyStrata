// src/poly/inkling2.cu - the kernels of Inkling 2's layers (inkling2_kernels.hpp).  The reference for the math is
// llama.cpp's inkling2 graph (src/models/inkling2.cpp, inkling2-loop.cpp).
//
// A token (or a token's head) a block, sums in a fixed order so that a token comes out the same in any company.
// The attention of a few tokens reads a key or a value with a warp, its lanes side by side: ink_attend().
// q81_div(): a block's scale divided as native_mmvq.cu's --use_fast_math division does it (q8_1_finite.hpp)
#define STRATA_Q81_FASTMATH_DIV 1
#include "inkling2_kernels.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::poly {
namespace {

inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }
using strata::kernels::launch_pdl;
using strata::kernels::pdl_trigger;
using strata::kernels::pdl_wait;
inline int grid(size_t n, int per) { return (int) ((n + (size_t) per - 1) / (size_t) per); }

// A launch the driver refused leaves its error for whoever asks next: ask here, so that it names its kernel.
void launched(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int WARP = 32;   // the lanes a shuffle spans: a warp, or half of a 64-lane wavefront

// The warp's sum of x, in every lane.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

struct QB {   // a q8_1 block, as native_quantize_q8_1() writes it
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");

// Value i of a row into its q8_1 blocks, native_quantize_q8_1_kernel's arithmetic; a warp's 32 lanes hold one block.
__device__ __forceinline__ void q8_put(QB* y, int i, float xi) {
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(strata::kernels::q81_div(amax, 127.0f));
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
    y[i / 32].qs[i % 32] = q;
    if (i % 32 == 0) y[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
}

// native_add_rms_norm_q8_1_kernel<blockDim>'s norm (no add) of a row x, a thread over its values, into y and q.
__device__ __forceinline__ void add_rms_q8(const float* x, const float* __restrict__ w, float* y, QB* q, int n,
                                           float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        ss = __fmaf_rn(v, v, ss);
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) sum = __fadd_rn(sum, part[k]);
    __syncthreads();
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
    for (int i = tid; i < n; i += blockDim.x) {
        const float xi = __fmul_rn(__fmul_rn(x[i], scale), w[i]);
        y[i] = xi;
        q8_put(q, i, xi);
    }
}

constexpr int kNormHeld = 4;   // the values a thread of a norm keeps: sum_threads() gives it at most 4 below 4,096

// add_rms_q8() of the H values a thread holds (value h is the row's threadIdx.x + h * blockDim.x), the same arithmetic
template <int H>
__device__ __forceinline__ void add_rms_q8_held(const float (&v)[H], const float* __restrict__ w, float* y, QB* q,
                                                int n, float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h)
        if (tid + h * int(blockDim.x) < n) ss = __fmaf_rn(v[h], v[h], ss);
    ss = warp_sum(ss);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) sum = __fadd_rn(sum, part[k]);
    __syncthreads();
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
#pragma unroll
    for (int h = 0; h < H; ++h) {
        const int i = tid + h * int(blockDim.x);
        if (i < n) {
            const float xi = __fmul_rn(__fmul_rn(v[h], scale), w[i]);
            y[i] = xi;
            q8_put(q, i, xi);
        }
    }
}

// The block's sum of x, the same in every thread: the warps' sums added in their order.  blockDim.x is a multiple
// of 32, and `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

constexpr int kQueries = 4;
constexpr int kMixThreads = 1024;
constexpr int kMixSums = 1 + kQueries;   // a state's sum of squares, and its product with each query

struct MixArgs {
    const float* s[InkStates::kMax];
    int n;
    const float* query;
    int nq;
    float* out[kQueries];
    const float* norm_w;   // with norm_out: out[0]'s rms_norm()
    float* norm_out;
    float eps;
    QB* q8;     // and the norm quantized (one row)
    int ffn;    // the norm is native_add_rms_norm_q8_1()'s
};

// A token a block.  A thread has every kMixThreads-th value of the row and reads each query's and each state's
// once: the states' squares and their products with the queries are summed over the warp, then over the warps in
// their order.
__device__ __forceinline__ void mix_row(const MixArgs& a, int n_embd) {
    __shared__ float part[InkStates::kMax * kMixSums][kMixThreads / WARP];
    __shared__ float whole[InkStates::kMax * kMixSums];
    __shared__ float coef[InkStates::kMax * kQueries];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float sum[InkStates::kMax][kMixSums];
#pragma unroll
    for (int s = 0; s < InkStates::kMax; ++s) {
#pragma unroll
        for (int k = 0; k < kMixSums; ++k) sum[s][k] = 0.0f;
    }
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float q[kQueries];
#pragma unroll
        for (int j = 0; j < kQueries; ++j) q[j] = j < a.nq ? a.query[size_t(j) * n_embd + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < InkStates::kMax; ++s) {
            if (s < a.n) {
                const float v = a.s[s][row + i];
                sum[s][0] += v * v;
#pragma unroll
                for (int j = 0; j < kQueries; ++j) sum[s][1 + j] += v * q[j];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < InkStates::kMax; ++s) {
#pragma unroll
        for (int k = 0; k < kMixSums; ++k) {
            if (s < a.n && k <= a.nq) {
                const float w = warp_sum(sum[s][k]);
                if (lane == 0) part[s * kMixSums + k][warp] = w;
            }
        }
    }
    __syncthreads();
    if (tid < a.n * kMixSums && tid % kMixSums <= a.nq) {
        float t = 0.0f;
        for (int w = 0; w < (int) blockDim.x / 32; ++w) t += part[tid][w];
        whole[tid] = t;
    }
    __syncthreads();
    if (tid < a.nq) {
        const int j = tid;
        const float sc = 1.0f / sqrtf((float) n_embd);
        float v[InkStates::kMax], inv[InkStates::kMax];
        float m = -INFINITY;
        for (int s = 0; s < a.n; ++s) {
            inv[s] = 1.0f / sqrtf(whole[s * kMixSums] / (float) n_embd + 1e-6f);
            v[s] = whole[s * kMixSums + 1 + j] * inv[s] * sc;
            m = fmaxf(m, v[s]);
        }
        float total = 0.0f;
        for (int s = 0; s < a.n; ++s) {
            v[s] = expf(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < a.n; ++s) coef[s * kQueries + j] = v[s] / total * inv[s];
    }
    __syncthreads();
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float x[InkStates::kMax];
#pragma unroll
        for (int s = 0; s < InkStates::kMax; ++s) x[s] = s < a.n ? a.s[s][row + i] : 0.0f;
#pragma unroll
        for (int j = 0; j < kQueries; ++j) {
            if (j < a.nq) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < InkStates::kMax; ++s) {
                    if (s < a.n) acc += coef[s * kQueries + j] * x[s];
                }
                a.out[j][row + i] = acc;
            }
        }
    }
    if (a.norm_out && a.ffn) {
        __shared__ float fpart[kMixThreads / WARP];
        add_rms_q8(a.out[0] + row, a.norm_w, a.norm_out + row, a.q8 + size_t(blockIdx.x) * (n_embd / 32), n_embd,
                   a.eps, fpart);
    } else if (a.norm_out) {   // rms_norm_kernel's arithmetic, a thread over the values it wrote
        __shared__ float npart[kMixThreads / WARP];
        float ss = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            const float v = a.out[0][row + i];
            ss += v * v;
        }
        ss = block_sum(ss, npart);
        const float scale = 1.0f / sqrtf(ss / float(n_embd) + a.eps);
        for (int i = tid; i < n_embd; i += blockDim.x) {
            const float v = a.out[0][row + i];
            const float y = a.norm_w ? v * scale * a.norm_w[i] : v * scale;
            a.norm_out[row + i] = y;
            if (a.q8) q8_put(a.q8 + size_t(blockIdx.x) * (n_embd / 32), i, y);
        }
    }
}

constexpr int kMixHeld = 3;   // the values of a row a thread of mix_held() holds: rows of up to 3 * kMixThreads

// mix_row() for NS states and NQ queries, a row of up to kMixHeld * kMixThreads values: a thread keeps the states'
// values it reads, and the outputs it writes, for what comes after (the outputs and the norm) instead of reading
// them again.  Every sum is mix_row()'s, term for term and in its order.
template <int NS, int NQ>
__device__ __forceinline__ void mix_held(const MixArgs& a, int n_embd) {
    __shared__ float part[InkStates::kMax * kMixSums][kMixThreads / WARP];
    __shared__ float whole[InkStates::kMax * kMixSums];
    __shared__ float coef[InkStates::kMax * kQueries];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float x[kMixHeld][NS];
    float sum[NS][1 + NQ];
#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k <= NQ; ++k) sum[s][k] = 0.0f;
    }
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            float q[NQ];
#pragma unroll
            for (int j = 0; j < NQ; ++j) q[j] = a.query[size_t(j) * n_embd + i];
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const float v = a.s[s][row + i];
                x[h][s] = v;
                sum[s][0] += v * v;
#pragma unroll
                for (int j = 0; j < NQ; ++j) sum[s][1 + j] += v * q[j];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k <= NQ; ++k) {
            const float w = warp_sum(sum[s][k]);
            if (lane == 0) part[s * kMixSums + k][warp] = w;
        }
    }
    __syncthreads();
    if (tid < NS * kMixSums && tid % kMixSums <= NQ) {
        float t = 0.0f;
        for (int w = 0; w < kMixThreads / WARP; ++w) t += part[tid][w];
        whole[tid] = t;
    }
    __syncthreads();
    if (tid < NQ) {
        const int j = tid;
        const float sc = 1.0f / sqrtf((float) n_embd);
        float v[NS], inv[NS];
        float m = -INFINITY;
        for (int s = 0; s < NS; ++s) {
            inv[s] = 1.0f / sqrtf(whole[s * kMixSums] / (float) n_embd + 1e-6f);
            v[s] = whole[s * kMixSums + 1 + j] * inv[s] * sc;
            m = fmaxf(m, v[s]);
        }
        float total = 0.0f;
        for (int s = 0; s < NS; ++s) {
            v[s] = expf(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < NS; ++s) coef[s * kQueries + j] = v[s] / total * inv[s];
    }
    __syncthreads();
    float out0[kMixHeld];
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < NS; ++s) acc += coef[s * kQueries + j] * x[h][s];
                a.out[j][row + i] = acc;
                if (j == 0) out0[h] = acc;
            }
        }
    }
    if (!a.norm_out) return;
    __shared__ float npart[kMixThreads / WARP];
    if (a.ffn) {   // add_rms_q8()'s arithmetic on the values written
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < kMixHeld; ++h)
            if (tid + h * kMixThreads < n_embd) ss = __fmaf_rn(out0[h], out0[h], ss);
        ss = warp_sum(ss);
        if (lane == 0) npart[warp] = ss;
        __syncthreads();
        float total = 0.0f;
        for (int k = 0; k < kMixThreads / WARP; ++k) total = __fadd_rn(total, npart[k]);
        const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(n_embd)), a.eps)));
        QB* q = a.q8 + size_t(blockIdx.x) * (n_embd / 32);
#pragma unroll
        for (int h = 0; h < kMixHeld; ++h) {
            const int i = tid + h * kMixThreads;
            if (i < n_embd) {
                const float xi = __fmul_rn(__fmul_rn(out0[h], scale), a.norm_w[i]);
                a.norm_out[row + i] = xi;
                q8_put(q, i, xi);
            }
        }
        return;
    }
    // rms_norm_kernel's arithmetic, a thread over the values it wrote
    float ss = 0.0f;
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h)
        if (tid + h * kMixThreads < n_embd) ss += out0[h] * out0[h];
    ss = block_sum(ss, npart);
    const float scale = 1.0f / sqrtf(ss / float(n_embd) + a.eps);
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            const float v = out0[h];
            const float y = a.norm_w ? v * scale * a.norm_w[i] : v * scale;
            a.norm_out[row + i] = y;
            if (a.q8) q8_put(a.q8 + size_t(blockIdx.x) * (n_embd / 32), i, y);
        }
    }
}

// mix_held() where it takes the row, else mix_row()
template <int NS, int NQ>
__device__ __forceinline__ void mix_any(const MixArgs& a, int n_embd) {
    if (n_embd <= kMixHeld * kMixThreads) mix_held<NS, NQ>(a, n_embd);
    else mix_row(a, n_embd);
}

template <int NS, int NQ>
__global__ void __launch_bounds__(kMixThreads) ink_mix_kernel(MixArgs a, int n_embd) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    mix_any<NS, NQ>(a, n_embd);
}

constexpr int kMixJobs = 12;
struct MixMany {
    MixArgs j[kMixJobs];
};

// blockIdx.y the job
template <int NS, int NQ>
__global__ void __launch_bounds__(kMixThreads) ink_mix_many_kernel(MixMany m, int n_embd) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    mix_any<NS, NQ>(m.j[blockIdx.y], n_embd);
}

// kernel<NS, NQ> for a call's n states and nq queries
template <template <int, int> class K, class... Args>
void launch_mix(int n, int nq, dim3 grid, cudaStream_t s, Args... args) {
    const auto go = [&](auto ns_c) {
        constexpr int NS = decltype(ns_c)::value;
        switch (nq) {
            case 1: K<NS, 1>::launch(grid, s, args...); break;
            case 2: K<NS, 2>::launch(grid, s, args...); break;
            case 3: K<NS, 3>::launch(grid, s, args...); break;
            default: K<NS, 4>::launch(grid, s, args...); break;
        }
    };
    switch (n) {
        case 1: go(std::integral_constant<int, 1>{}); break;
        case 2: go(std::integral_constant<int, 2>{}); break;
        case 3: go(std::integral_constant<int, 3>{}); break;
        case 4: go(std::integral_constant<int, 4>{}); break;
        case 5: go(std::integral_constant<int, 5>{}); break;
        default: go(std::integral_constant<int, 6>{}); break;
    }
}

template <int NS, int NQ>
struct MixOne {
    static void launch(dim3 grid, cudaStream_t s, const MixArgs& a, int n_embd) {
        launch_pdl(ink_mix_kernel<NS, NQ>, dim3(grid), dim3(kMixThreads), 0, s, a, n_embd);
    }
};

template <int NS, int NQ>
struct MixJobs {
    static void launch(dim3 grid, cudaStream_t s, const MixMany& m, int n_embd) {
        launch_pdl(ink_mix_many_kernel<NS, NQ>, dim3(grid), dim3(kMixThreads), 0, s, m, n_embd);
    }
};

constexpr int kSet = 3;   // the controllers one launch takes

struct CatNormSet {
    const float* cur[kSet];
    const float* msg[kSet];
    float* out[kSet];
};

constexpr int kCatThreads = 256;   // ink_cat_norm_kernel's block

// Thread tid's shares of the squares of a row of cur and of msg, in ink_cat_norm_kernel's block: every
// kCatThreads-th value from tid.
__device__ __forceinline__ void cat_squares(const float* cur, const float* msg, int n_embd,
                                            int tid, float& a, float& b) {
    a = 0.0f, b = 0.0f;
    for (int i = tid; i < n_embd; i += kCatThreads) {
        a += cur[i] * cur[i];
        b += msg[i] * msg[i];
    }
}

__device__ __forceinline__ float cat_scale(float sum, int n_embd) { return 1.0f / sqrtf(sum / (float) n_embd + 1e-6f); }

// A token's pair a block, blockIdx.y the pair.
__global__ void ink_cat_norm_kernel(CatNormSet s, int n_embd) {
    __shared__ float part[kCatThreads / WARP];
    const float* __restrict__ cur = s.cur[blockIdx.y];
    const float* __restrict__ msg = s.msg[blockIdx.y];
    float* __restrict__ out = s.out[blockIdx.y];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x;
    float a, b;
    cat_squares(cur + row, msg + row, n_embd, tid, a, b);
    const float ia = cat_scale(block_sum(a, part), n_embd);
    const float ib = cat_scale(block_sum(b, part), n_embd);
    float* o = out + 2 * row;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        o[i] = cur[row + i] * ia;
        o[n_embd + i] = msg[row + i] * ib;
    }
}

struct CtrlDown {
    const float* cur[kSet];
    const float* msg[kSet];
    const float* w[kSet];   // [2 * n_embd] an output
    float* y[kSet];
};

constexpr int kDownThreads = 512;   // ink_ctrl_down_kernel's block: half of gemv_set_kernel's

// ink_cat_norm_kernel and then gemv_set_kernel's product with the controller's down matrix in one launch: output o
// of controller j (blockIdx.x = j * rank + o) of token blockIdx.y.  Its first kCatThreads threads make the two norms
// as ink_cat_norm_kernel's block does, and the product takes each normed value as that kernel writes it.  The
// product is gemv_set_kernel's block of 1024 threads (2 * n_embd values) in kDownThreads: thread t has the sums of
// its threads t and t + 512, warps w and w + 16, and the 32 warps' sums are added in their order.  The same bits,
// and with half the threads a block, a launch's 3 * 64 blocks are one wave on the card.
__global__ void __launch_bounds__(kDownThreads) ink_ctrl_down_kernel(CtrlDown d, int n_embd, int rank) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float pa[kCatThreads / WARP], pb[kCatThreads / WARP], part[32];
    const int j = blockIdx.x / rank, o = blockIdx.x % rank, tid = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * n_embd;
    const float* cur = d.cur[j] + row;
    const float* msg = d.msg[j] + row;
    if (tid < kCatThreads) {
        float a, b;
        cat_squares(cur, msg, n_embd, tid, a, b);
        a = warp_sum(a);
        b = warp_sum(b);
        if ((tid & 31) == 0) pa[tid >> 5] = a, pb[tid >> 5] = b;
    }
    __syncthreads();
    float sa = 0.0f, sb = 0.0f;
    for (int k = 0; k < kCatThreads / WARP; ++k) sa += pa[k];
    for (int k = 0; k < kCatThreads / WARP; ++k) sb += pb[k];
    const float ia = cat_scale(sa, n_embd), ib = cat_scale(sb, n_embd);
    const float* __restrict__ wr = d.w[j] + size_t(o) * 2 * n_embd;
    const int n_in = 2 * n_embd;
    float acc0 = 0.0f, acc1 = 0.0f;   // gemv_set_kernel's threads tid and tid + kDownThreads
    for (int i = tid; i < n_in; i += 2 * kDownThreads) {
        const float x0 = i < n_embd ? cur[i] * ia : msg[i - n_embd] * ib;
        acc0 += wr[i] * x0;
        const int i1 = i + kDownThreads;
        if (i1 < n_in) {
            const float x1 = i1 < n_embd ? cur[i1] * ia : msg[i1 - n_embd] * ib;
            acc1 += wr[i1] * x1;
        }
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if ((tid & 31) == 0) part[tid >> 5] = acc0, part[(tid >> 5) + kDownThreads / WARP] = acc1;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 2 * kDownThreads / WARP; ++k) sum += part[k];
        d.y[j][size_t(blockIdx.y) * rank + o] = sum;
    }
}

struct GateSet {
    const float* cur[kSet];
    const float* msg[kSet];
    const float* gain[kSet];
    const float* forget[kSet];
    float* out[kSet];
    int count;
};

// A value a thread, of every controller of the set: all its cur are read before an out is written.
__global__ void ink_gate_kernel(GateSet s, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c[kSet];
#pragma unroll
    for (int j = 0; j < kSet; ++j)
        if (j < s.count) c[j] = s.cur[j][i];
#pragma unroll
    for (int j = 0; j < kSet; ++j)
        if (j < s.count) s.out[j][i] = c[j] - tanhf(s.forget[j][i]) * c[j] + tanhf(s.gain[j][i]) * s.msg[j][i];
}

// Token t's convolved value of channel c (ink_conv()).
__device__ __forceinline__ float conv_value(const float* x, const float* state, const float* __restrict__ w,
                                            const float* dyn, int channels,
                                            int heads, int taps, int t, int c, float norm) {
    float past[8];   // past[k]: the input k tokens back
    for (int k = 0; k < taps; ++k) {
        const int u = t - k;
        past[k] = u >= 0 ? x[size_t(u) * channels + c] : state[size_t(taps - 1 + u) * channels + c];
    }
    float acc = 0.0f;
    for (int j = 0; j < taps; ++j) acc += past[taps - 1 - j] * w[size_t(c) * taps + j];
    if (dyn) {
        const int head = c / (channels / heads);
        const float* d = dyn + (size_t(t) * heads + head) * taps;
        for (int lag = 0; lag < taps; ++lag) acc += past[lag] * d[lag];
    }
    return (past[0] + acc) * norm;
}

// Channel c's state after a call of `rows` tokens: the last taps - 1 inputs, oldest first.
__device__ __forceinline__ void conv_state_at(const float* x, float* state, int channels, int taps,
                                              int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        state[size_t(k) * channels + c] =
            u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
    }
}

// Channel c's state after the first `rows` tokens of a call (conv_state_at()'s), from the state before the call into
// out, which may be elsewhere.
__device__ __forceinline__ void conv_state_into(const float* x, const float* state, float* out, int channels, int taps,
                                                int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        out[size_t(k) * channels + c] = u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
    }
}

// One head's clip (ink_clip()), the whole block taking part; p is the token's position.
__device__ __forceinline__ void clip_head(float* v, const float* __restrict__ sc, float thr, int dim, float alpha,
                                          float floor_, long long p, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    const float tot = block_sum(ss, part);
    float f = fminf(thr / (sqrtf(tot) + 1e-6f), 1.0f);
    if (alpha > 0.0f) f *= 1.0f + alpha * logf(fmaxf(1.0f, (float) (p + 1) / floor_));
    for (int i = tid; i < dim; i += blockDim.x) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

// One head's RMS (ink_head_rms()), the whole block taking part.
__device__ __forceinline__ void rms_head(float* v, int dim, float eps, float* part) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) ss += v[i] * v[i];
    const float f = 1.0f / sqrtf(block_sum(ss, part) / (float) dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) v[i] *= f;
}

__global__ void ink_conv_kernel(const float* __restrict__ x, const float* __restrict__ state,
                                const float* __restrict__ w, const float* __restrict__ dyn, float* __restrict__ out,
                                int channels, int heads, int taps, int rows, float norm) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * channels) return;
    const int t = (int) (idx / channels), c = (int) (idx % channels);
    out[idx] = conv_value(x, state, w, dyn, channels, heads, taps, t, c, norm);
}

// A channel a thread: the state after each token of the call that bits names (bit t: after token t), the j-th such
// state at snap + j * stride.  Before ink_conv_state_kernel, which overwrites the state.
__global__ void ink_conv_snaps_kernel(const float* x, const float* state, float* snap, size_t stride, unsigned bits,
                                      int channels, int taps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int j = 0;
    for (unsigned b = bits; b; b &= b - 1, ++j)
        conv_state_into(x, state, snap + size_t(j) * stride, channels, taps, __ffs(b), c);
}

// a channel a thread
__global__ void ink_conv_state_kernel(const float* __restrict__ x, float* state, int channels, int taps, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    conv_state_at(x, state, channels, taps, rows, c);
}

// a token's head a block
__global__ void ink_clip_kernel(float* x, const float* __restrict__ scale, float thr, int dim, int heads,
                                long long pos, float alpha, float floor_, const long long* __restrict__ at) {
    __shared__ float part[128 / WARP];
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    clip_head(x + (size_t(row) * heads + head) * dim, scale ? scale + size_t(head) * dim : nullptr, thr, dim, alpha,
              floor_, at ? at[2 * row] : pos + row, part);
}

__global__ void ink_head_rms_kernel(float* x, int dim, int heads, float eps) {
    __shared__ float part[128 / WARP];
    rms_head(x + size_t(blockIdx.x) * dim, dim, eps, part);
}

__global__ void ink_conv_add_kernel(const float* x, float* state, const float* __restrict__ w,
                                    float* __restrict__ out, float* acc, float* keep, float a, int mode, int channels,
                                    int taps, float norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float y = conv_value(x, state, w, nullptr, channels, 1, taps, 0, c, norm);
    out[c] = y;
    conv_state_at(x, state, channels, taps, 1, c);
    if (mode == 0) {
        acc[c] = acc[c] + y * a;
    } else {
        if (mode == 1) keep[c] = acc[c];
        acc[c] = y;
    }
}

__global__ void ink_moe_conv_add_kernel(const float* x, const float* w, const float* __restrict__ scale,
                                        const int* ids, int n_used,
                                        float* fo, float* state, const float* __restrict__ cw, float* __restrict__ out,
                                        float* acc, float a, int channels, int taps, float norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float s = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        const float wk = scale ? w[k] * scale[ids[k]] : w[k];   // expert_weights_kernel's product
        s += wk * x[size_t(k) * channels + i];
    }
    fo[i] = 0.0f + s;   // moe_sum() onto the zeros fo was
    const float y = conv_value(fo, state, cw, nullptr, channels, 1, taps, 0, i, norm);
    out[i] = y;
    conv_state_at(fo, state, channels, taps, 1, i);
    acc[i] = acc[i] + y * a;
}

struct QkvArgs {
    InkQkv a;
    float k_norm, v_norm;
};

// ink_qkv(): blocks [0, n_head) Q's heads, then K's n_kv, V's n_kv, and the relative positions' n_head; 128 threads.
__global__ void ink_qkv_kernel(QkvArgs g) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[128 / WARP];
    const InkQkv& a = g.a;
    const long long p = a.at ? a.at[0] : (long long) a.pos;
    int b = blockIdx.x;
    if (b < a.n_head) {
        clip_head(a.q + size_t(b) * a.dim, a.q_scale ? a.q_scale + size_t(b) * a.dim : nullptr, a.qk_thr, a.dim,
                  a.alpha, a.floor, p, part);
        return;
    }
    b -= a.n_head;
    if (b < 2 * a.n_kv) {
        const bool is_v = b >= a.n_kv;
        const int h = is_v ? b - a.n_kv : b, channels = a.n_kv * a.dim, i = threadIdx.x;
        const float* x = is_v ? a.vraw : a.kraw;
        float* state = is_v ? a.vstate : a.kstate;
        float* out = is_v ? a.v : a.k;
        if (i < a.dim) {
            const int c = h * a.dim + i;
            out[c] = conv_value(x, state, is_v ? a.vw : a.kw, is_v ? a.vdyn : a.kdyn, channels, a.n_kv, a.taps, 0, c,
                                is_v ? g.v_norm : g.k_norm);
            conv_state_at(x, state, channels, a.taps, 1, c);
        }
        float* head = out + size_t(h) * a.dim;
        if (is_v) rms_head(head, a.dim, a.eps, part);
        else clip_head(head, nullptr, a.qk_thr, a.dim, 0.0f, 1.0f, p, part);
        if (i < a.dim) {
            const long long row = a.ring > 0 ? p % a.ring : p;
            ((__half*) (is_v ? a.vals : a.keys))[size_t(row) * channels + size_t(h) * a.dim + i] = __float2half(head[i]);
        }
        return;
    }
    b -= 2 * a.n_kv;
    clip_head(a.r + size_t(b) * a.n_rel, a.r_scale ? a.r_scale + size_t(b) * a.n_rel : nullptr, a.rel_thr, a.n_rel,
              a.alpha, a.floor, p, part);
}

__global__ void ink_mul_sigmoid_kernel(float* x, const float* __restrict__ gate, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

__global__ void ink_store_ring_kernel(const float* __restrict__ x, __half* __restrict__ cache, int n, long long pos,
                                      int rows, int ring) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * n) return;
    const long long t = (long long) (idx / n);
    cache[size_t((pos + t) % ring) * n + idx % n] = __float2half(x[idx]);
}

__global__ void ink_softcap_kernel(float* logits, int n_keep, int n_all, int rows, float cap) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * n_all) return;
    logits[idx] = (int) (idx % n_all) < n_keep ? cap * tanhf(logits[idx] / cap) : -INFINITY;
}

constexpr int kAttnThreads = 1024;   // a block of the softmax kernel
constexpr int kMaxPer = 4;           // the most query heads on one key head
constexpr int kLaneHalves = 4;       // the most halves of a cache row's head a lane holds: heads of up to 128 values
constexpr int kScoreWarps = 8;       // a block of the score kernel
constexpr int kScoreKeys = 4;        // the keys a warp of it takes, one after the other
constexpr int kValueKeys = 8;        // the same for the value kernel: a lane has a weight of the warp's each
constexpr size_t kValueShared = size_t(48) << 10;   // the shared memory a block has without asking for more
static_assert(kMaxPer * kValueKeys <= 32, "a warp's weights are a lane each");

// A lane's share of a head's row of the cache, which the warp reads side by side: n halves (2 or 4) of the row's
// 32 * n, as floats.  The rest of out is 0.
__device__ __forceinline__ void lane_halves(const __half* head, int lane, int n, float* out) {
    const __half2* p = (const __half2*) (head + lane * n);
#pragma unroll
    for (int c = 0; c < kLaneHalves / 2; ++c) {
        float2 f = make_float2(0.0f, 0.0f);
        if (2 * c < n) f = __half22float2(p[c]);
        out[2 * c] = f.x;
        out[2 * c + 1] = f.y;
    }
}

// The scores of a token's heads on one key head (blockIdx.x) against a block of keys (blockIdx.y).  A warp has
// kScoreKeys keys, one at a time: its lanes read the key side by side, once for all of the group's heads, and
// their products with the queries are summed across the warp.  The bias's products are the first n_rel lanes'.
// t0 is the launch's first token among the call's, nk_max the scores a head has room for.
__global__ void ink_score_kernel(InkAttn a, int t0, int nk_max) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float qs[kMaxPer * 128];
    __shared__ float rs[kMaxPer * 32];
    __shared__ float sinks[kMaxPer];
    const int per = a.n_head / a.n_kv, lh = a.dim / 32;
    const int tt = blockIdx.x / a.n_kv, g = blockIdx.x % a.n_kv, t = t0 + tt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    if ((int) (blockIdx.y * kScoreWarps * kScoreKeys) >= nk) return;   // (the whole block)
    const size_t head0 = size_t(t) * a.n_head + size_t(g) * per;       // the group's heads lie together
    for (int i = tid; i < per * a.dim; i += blockDim.x) qs[i] = a.q[head0 * a.dim + i] * a.scale;
    for (int i = tid; i < per * a.n_rel; i += blockDim.x) rs[i] = a.r[head0 * a.n_rel + i];
    if (tid < per) sinks[tid] = a.sink[head0 + tid];
    __syncthreads();
    const int j0 = (blockIdx.y * kScoreWarps + warp) * kScoreKeys, j1 = j0 + kScoreKeys < nk ? j0 + kScoreKeys : nk;
    if (j0 >= nk) return;   // (a whole warp)
    float q[kMaxPer][kLaneHalves], r[kMaxPer];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) q[hp][c] = hp < per && c < lh ? qs[hp * a.dim + lane * lh + c] : 0.0f;
        r[hp] = hp < per && lane < a.n_rel ? rs[hp * a.n_rel + lane] : 0.0f;
    }
    float* s = a.score + (size_t(tt) * a.n_head + size_t(g) * per) * nk_max;
    // the warp's keys and their bias values first, so that their reads are on their way together (the scores'
    // stores between them could be to the same memory, as far as the compiler knows)
    float k[kScoreKeys][kLaneHalves], rb[kScoreKeys];
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
        if (j0 + u < j1) {
            lane_halves((const __half*) a.keys + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, k[u]);
            const long long dist = p - (lo + j0 + u);
            rb[u] = lane < a.n_rel && dist < a.extent ? a.rel_bias[size_t(dist) * a.n_rel + lane] : 0.0f;
            if (++row == a.ring) row = 0;
        }
    }
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
        if (j0 + u < j1) {
#pragma unroll
            for (int hp = 0; hp < kMaxPer; ++hp) {
                if (hp < per) {
                    float d = rb[u] * r[hp];
#pragma unroll
                    for (int c = 0; c < kLaneHalves; ++c) d += q[hp][c] * k[u][c];
                    d = warp_sum(d);
                    if (lane == 0) s[size_t(hp) * nk_max + j0 + u] = d - sinks[hp];
                }
            }
        }
    }
}

// A token's head a block: the scores' softmax with its extra term of score 0.  The scores become the weights'
// numerators (exp against the largest), and norm gets what they are divided by, a token's head each.
__global__ void ink_softmax_kernel(InkAttn a, int t0, int nk_max, float* norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[kAttnThreads / WARP];
    const int tt = blockIdx.x / a.n_head, h = blockIdx.x % a.n_head, t = t0 + tt, tid = threadIdx.x;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    float* s = a.score + (size_t(tt) * a.n_head + h) * nk_max;
    float m = -INFINITY;
    for (int j = tid; j < nk; j += blockDim.x) m = fmaxf(m, s[j]);
    m = fmaxf(block_max(m, part), 0.0f);
    float sum = 0.0f;
    for (int j = tid; j < nk; j += blockDim.x) {
        s[j] = expf(s[j] - m);
        sum += s[j];
    }
    sum = block_sum(sum, part);
    if (tid == 0) norm[blockIdx.x] = sum + expf(-m);
}

// The values' sums under the weights for a token's heads on one key head (blockIdx.x), a block of keys
// (blockIdx.y), into parts: `dim` values a head and a block.  A warp has kValueKeys values, one at a time, read
// as the score kernel reads a key; a lane holds one of the warp's weights, a head and a key each.  The warps'
// sums are added in their order.  value_warps() launches up to 32 warps, so the registers must fit 1,024 threads on
// every card: sm_86 and sm_89 gave this kernel 66 a thread without the bound, and the launch failed there.
__global__ void __launch_bounds__(1024) ink_value_kernel(InkAttn a, int t0, int nk_max, float* parts, int n_parts) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    extern __shared__ float sums[];   // a warp's, a head of the group: dim
    const int per = a.n_head / a.n_kv, lh = a.dim / 32;
    const int tt = blockIdx.x / a.n_kv, g = blockIdx.x % a.n_kv, t = t0 + tt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x / 32;
    const long long p = a.at ? a.at[2 * t] : a.pos + t;
    const long long lo = a.window > 0 ? (p - a.window + 1 > 0 ? p - a.window + 1 : 0) : 0;
    const int nk = (int) (p - lo + 1);
    const int j0 = (blockIdx.y * warps + warp) * kValueKeys, j1 = j0 + kValueKeys < nk ? j0 + kValueKeys : nk;
    const size_t head0 = size_t(tt) * a.n_head + size_t(g) * per;
    const float* s = a.score + head0 * nk_max;
    const int mine = j0 + lane % kValueKeys;   // the key of this lane's weight, for head lane / kValueKeys
    const float weight = lane < per * kValueKeys && mine < nk ? s[size_t(lane / kValueKeys) * nk_max + mine] : 0.0f;
    float acc[kMaxPer][kLaneHalves];
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) acc[hp][c] = 0.0f;
    }
    // the warp's values first, so that their reads are on their way together; then each in its turn
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
    float v[kValueKeys][kLaneHalves];
#pragma unroll
    for (int u = 0; u < kValueKeys; ++u) {
        if (j0 + u < j1) {
            lane_halves((const __half*) a.vals + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, v[u]);
            if (++row == a.ring) row = 0;
        }
    }
#pragma unroll
    for (int u = 0; u < kValueKeys; ++u) {
        if (j0 + u < j1) {
#pragma unroll
            for (int hp = 0; hp < kMaxPer; ++hp) {
                if (hp < per) {
                    const float w = __shfl_sync(0xffffffff, weight, hp * kValueKeys + u, WARP);
#pragma unroll
                    for (int c = 0; c < kLaneHalves; ++c) acc[hp][c] += w * v[u][c];
                }
            }
        }
    }
#pragma unroll
    for (int hp = 0; hp < kMaxPer; ++hp) {
#pragma unroll
        for (int c = 0; c < kLaneHalves; ++c) {
            if (hp < per && c < lh) sums[(size_t(warp) * per + hp) * a.dim + lane * lh + c] = acc[hp][c];
        }
    }
    __syncthreads();
    for (int e = tid; e < per * a.dim; e += blockDim.x) {   // value e % dim of head e / dim
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) total += sums[size_t(w) * per * a.dim + e];
        parts[((head0 + e / a.dim) * n_parts + blockIdx.y) * a.dim + e % a.dim] = total;
    }
}

// A token's head a block: its parts added in their order, over its softmax's norm.
__global__ void ink_join_kernel(InkAttn a, int t0, const float* parts, int n_parts, const float* norm) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int tt = blockIdx.x / a.n_head, h = blockIdx.x % a.n_head, t = t0 + tt;
    const float inv = 1.0f / norm[blockIdx.x];
    const float* pt = parts + size_t(blockIdx.x) * n_parts * a.dim;
    for (int i = threadIdx.x; i < a.dim; i += blockDim.x) {
        float total = 0.0f;
        for (int k = 0; k < n_parts; ++k) total += pt[size_t(k) * a.dim + i];
        const size_t at = (size_t(t) * a.n_head + h) * a.dim + i;
        float v = total * inv;
        if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at]));   // ink_mul_sigmoid_kernel's
        a.out[at] = v;
        if (a.q8) q8_put((QB*) a.q8, (int) at, v);   // one token's row, for the output product
    }
}

__device__ __forceinline__ float2 situ_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

// with q8: the rows also quantized, back to back as one vector (n a multiple of 32: a warp's values are a block)
__global__ void situ_rows_kernel(const float* gate_up, float* __restrict__ out, int n, float beta,
                                 float lb, ExpertScales s, QB* q8) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = situ_scales(s, blockIdx.y);
    const float g = row[i] * sc.x;
    float u = row[n + i] * sc.y;
    if (lb > 0.0f) u = lb * tanhf(u / lb);
    const float h = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
    out[size_t(blockIdx.y) * n + i] = h;
    if (q8) q8_put(q8, (int) (size_t(blockIdx.y) * n + i), h);
}

struct Control3 {
    const float* cur[3];
    const float* msg[3];
    const float* ch[3];
    const float* gw[3];
    const float* fw[3];
    float* out[3];
};

// a warp value i of the three controllers: 6 products of rank values (gemv_set_warp_kernel's), then the gates
__global__ void ink_control3_gate_kernel(Control3 c, int n, int rank) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    const int i = int(blockIdx.x) * 8 + int(threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= n) return;
    float g[3], f[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float* gr = c.gw[j] + size_t(i) * rank;
        const float* fr = c.fw[j] + size_t(i) * rank;
        float ga = 0.0f, fa = 0.0f;
        for (int k = lane; k < rank; k += 32) {
            ga += gr[k] * c.ch[j][k];
            fa += fr[k] * c.ch[j][k];
        }
        g[j] = 0.0f + warp_sum(ga);
        f[j] = 0.0f + warp_sum(fa);
    }
    if (lane != 0) return;
    float cv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) cv[j] = c.cur[j][i];
#pragma unroll
    for (int j = 0; j < 3; ++j) c.out[j][i] = cv[j] - tanhf(f[j]) * cv[j] + tanhf(g[j]) * c.msg[j][i];
}

}  // namespace

struct NormMany {
    const float* x[3];
    const float* w[3];
    float* y[3];
    QB* q[3];
};

// rms_norm_many_kernel's row (blockIdx.y the input), each value also quantized
__global__ void ink_norm_many_q8_kernel(NormMany m, int n, float eps) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[32];
    const float* xr = m.x[blockIdx.y];
    const float* __restrict__ w = m.w[blockIdx.y];
    float* yr = m.y[blockIdx.y];
    if (n <= kNormHeld * int(blockDim.x)) {   // the values read once and kept for the norm
        float v[kNormHeld];
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                v[h] = xr[i];
                ss += v[h] * v[h];
            }
        }
        ss = block_sum(ss, part);
        const float scale = 1.0f / sqrtf(ss / float(n) + eps);
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                const float y = w ? v[h] * scale * w[i] : v[h] * scale;
                yr[i] = y;
                q8_put(m.q[blockIdx.y], i, y);
            }
        }
        return;
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = w ? xr[i] * scale * w[i] : xr[i] * scale;
        yr[i] = y;
        q8_put(m.q[blockIdx.y], i, y);
    }
}

// ink_gate_kernel's values of one row, then add_rms_q8() of them
__global__ void ink_gate_norm_q8_kernel(const float* cur, const float* msg, const float* gain, const float* forget,
                                        float* out, const float* __restrict__ w,
                                        float* y, QB* q, int n, float eps) {
    pdl_trigger();   // PDL (pdl.hpp): the launch after this one may start
    pdl_wait();
    __shared__ float part[32];
    if (n <= kNormHeld * int(blockDim.x)) {   // the gated values kept for the norm
        float o[kNormHeld];
#pragma unroll
        for (int h = 0; h < kNormHeld; ++h) {
            const int i = threadIdx.x + h * blockDim.x;
            if (i < n) {
                const float c = cur[i];
                o[h] = c - tanhf(forget[i]) * c + tanhf(gain[i]) * msg[i];
                out[i] = o[h];
            }
        }
        add_rms_q8_held(o, w, y, q, n, eps, part);
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float c = cur[i];
        out[i] = c - tanhf(forget[i]) * c + tanhf(gain[i]) * msg[i];
    }
    add_rms_q8(out, w, y, q, n, eps, part);
}

void ink_norm_many_q8(const float* const* x, const float* const* w, float* const* y, void* const* q8, int count, int n,
                      float eps, void* stream) {
    if (count < 1 || count > 3 || n % 32 != 0) throw std::invalid_argument("ink_norm_many_q8: 1 to 3 rows of 32k");
    NormMany m{};
    for (int j = 0; j < count; ++j) {
        m.x[j] = x[j];
        m.w[j] = w[j];
        m.y[j] = y[j];
        m.q[j] = (QB*) q8[j];
    }
    launch_pdl(ink_norm_many_q8_kernel, dim3(1, count), dim3(sum_threads(n)), 0, cs(stream), m, n, eps);
    launched("ink_norm_many_q8");
}

void ink_gate_norm_q8(const float* cur, const float* msg, const float* gain, const float* forget, float* out,
                      const float* w, float* y, void* q8, int n, float eps, void* stream) {
    const int threads = sum_threads(n);
    if (n % 32 != 0 || threads != std::min(1024, std::max(threads, n)))   // add_rms_norm_q8_1's block is this one
        throw std::invalid_argument("ink_gate_norm_q8: not add_rms_norm_q8_1's block");
    launch_pdl(ink_gate_norm_q8_kernel, dim3(1), dim3(threads), 0, cs(stream), cur, msg, gain, forget, out, w, y,
               (QB*) q8, n, eps);
    launched("ink_gate_norm_q8");
}

void ink_mix_norm(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
                  const float* norm_w, float* norm_out, float eps, void* stream, void* q8, bool ffn) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > kQueries)
        throw std::invalid_argument("ink_mix: 1 to 6 states and 1 to 4 queries");
    if (rows <= 0) return;
    if (norm_out && sum_threads(n_embd) != kMixThreads)   // rms_norm()'s block must be this one
        throw std::invalid_argument("ink_mix_norm: a row of this width is normed by another block");
    MixArgs a{};
    for (int s = 0; s < states.n; ++s) a.s[s] = states.p[s];
    a.n = states.n;
    a.query = query;
    a.nq = n_query;
    for (int j = 0; j < n_query; ++j) a.out[j] = out[j];
    a.norm_w = norm_w;
    a.norm_out = norm_out;
    a.eps = eps;
    a.q8 = (QB*) q8;
    a.ffn = ffn;
    if ((q8 || ffn) && (rows != 1 || !norm_out || (ffn && (!q8 || !norm_w))))
        throw std::invalid_argument("ink_mix_norm: a quantized norm is one row's");
    launch_mix<MixOne>(a.n, a.nq, dim3(rows), cs(stream), a, n_embd);
    launched("ink_mix");
}

void ink_mix_many(const InkStates& states, const float* const* query, int n_query, float* const* out, int jobs,
                  int n_embd, void* stream) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > kQueries || jobs < 1 || jobs > kMixJobs)
        throw std::invalid_argument("ink_mix_many: 1 to 6 states, 1 to 4 queries, 1 to 12 jobs");
    MixMany m{};
    for (int k = 0; k < jobs; ++k) {
        MixArgs& a = m.j[k];
        for (int s = 0; s < states.n; ++s) a.s[s] = states.p[s];
        a.n = states.n;
        a.query = query[k];
        a.nq = n_query;
        for (int j = 0; j < n_query; ++j) a.out[j] = out[k * n_query + j];
    }
    launch_mix<MixJobs>(states.n, n_query, dim3(1, jobs), cs(stream), m, n_embd);
    launched("ink_mix_many");
}

void ink_mix(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
             void* stream) {
    ink_mix_norm(states, query, n_query, out, n_embd, rows, nullptr, nullptr, 0.f, stream);
}

void ink_cat_norm_set(const float* const* cur, const float* const* msg, float* const* out, int count, int n_embd,
                      int rows, void* stream) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_cat_norm_set: 1 to 3 pairs");
    if (rows <= 0) return;
    CatNormSet s{};
    for (int j = 0; j < count; ++j) {
        s.cur[j] = cur[j];
        s.msg[j] = msg[j];
        s.out[j] = out[j];
    }
    ink_cat_norm_kernel<<<dim3(rows, count), 256, 0, cs(stream)>>>(s, n_embd);
    launched("ink_cat_norm");
}

void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream) {
    ink_cat_norm_set(&cur, &msg, &out, 1, n_embd, rows, stream);
}

bool ink_ctrl_down(const float* const* cur, const float* const* msg, const float* const* down, float* const* ch,
                   int count, int n_embd, int rank, int rows, void* stream) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_ctrl_down: 1 to 3 controllers");
    if (sum_threads(2 * n_embd) != 1024) return false;   // gemv_set_kernel's block is another
    if (rows <= 0) return true;
    CtrlDown d{};
    for (int j = 0; j < count; ++j) {
        d.cur[j] = cur[j];
        d.msg[j] = msg[j];
        d.w[j] = down[j];
        d.y[j] = ch[j];
    }
    launch_pdl(ink_ctrl_down_kernel, dim3(count * rank, rows), dim3(kDownThreads), 0, cs(stream), d, n_embd, rank);
    launched("ink_ctrl_down");
    return true;
}

void ink_gate_set(const float* const* cur, const float* const* msg, const float* const* gain,
                  const float* const* forget, float* const* out, int count, size_t n, void* stream) {
    if (count < 1 || count > kSet) throw std::invalid_argument("ink_gate_set: 1 to 3 controllers");
    if (n == 0) return;
    GateSet s{};
    for (int j = 0; j < count; ++j) {
        s.cur[j] = cur[j];
        s.msg[j] = msg[j];
        s.gain[j] = gain[j];
        s.forget[j] = forget[j];
        s.out[j] = out[j];
    }
    s.count = count;
    ink_gate_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(s, n);
    launched("ink_gate");
}

void ink_gate(const float* cur, const float* msg, const float* gain, const float* forget, float* out, size_t n,
              void* stream) {
    ink_gate_set(&cur, &msg, &gain, &forget, &out, 1, n, stream);
}

void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels, int heads,
              int taps, int rows, void* stream, float* snap, size_t snap_stride, unsigned snap_bits) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv: 2 to 8 taps");
    if (rows <= 0) return;
    const float norm = 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * taps + 1));
    ink_conv_kernel<<<grid((size_t) rows * channels, 256), 256, 0, cs(stream)>>>(x, state, w, dyn, out, channels,
                                                                                  heads, taps, rows, norm);
    launched("ink_conv");
    if (snap && snap_bits) {
        ink_conv_snaps_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, snap, snap_stride,
                                                                                    snap_bits, channels, taps);
        launched("ink_conv_snaps");
    }
    ink_conv_state_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, channels, taps, rows);
    launched("ink_conv_state");
}

void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    launch_pdl(ink_conv_add_kernel, dim3(grid((size_t) channels, 256)), dim3(256), 0, cs(stream), x, state, w, out,
               acc, keep, a, mode, channels, taps, norm);
    launched("ink_conv_add");
}

void ink_moe_conv_add(const float* x, const float* w, const float* scale, const int* ids, int n_used, float* fo,
                      float* state, const float* cw, float* out, float* acc, float a, int channels, int taps,
                      void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_moe_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    launch_pdl(ink_moe_conv_add_kernel, dim3(grid((size_t) channels, 256)), dim3(256), 0, cs(stream), x, w, scale, ids,
               n_used, fo, state, cw, out, acc, a, channels, taps, norm);
    launched("ink_moe_conv_add");
}

void ink_qkv(const InkQkv& a, void* stream) {
    if (a.dim > 128 || a.n_rel > 128 || a.taps < 2 || a.taps > 8 || a.n_kv < 1)
        throw std::invalid_argument("ink_qkv: heads of up to 128 values, 2 to 8 taps");
    QkvArgs g{a, 1.0f / std::sqrt((float) ((a.kdyn ? 2 : 1) * a.taps + 1)),
              1.0f / std::sqrt((float) ((a.vdyn ? 2 : 1) * a.taps + 1))};
    launch_pdl(ink_qkv_kernel, dim3(2 * a.n_head + 2 * a.n_kv), dim3(128), 0, cs(stream), g);
    launched("ink_qkv");
}

void ink_clip(float* x, const float* scale, float thr, int dim, int heads, int rows, int64_t pos, float alpha,
              float floor, void* stream, const long long* at) {
    if (rows <= 0) return;
    ink_clip_kernel<<<rows * heads, 128, 0, cs(stream)>>>(x, scale, thr, dim, heads, (long long) pos, alpha, floor,
                                                          at);
    launched("ink_clip");
}

void ink_head_rms(float* x, int dim, int heads, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    ink_head_rms_kernel<<<rows * heads, 128, 0, cs(stream)>>>(x, dim, heads, eps);
    launched("ink_head_rms");
}

void ink_mul_sigmoid(float* x, const float* gate, size_t n, void* stream) {
    if (n == 0) return;
    ink_mul_sigmoid_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, gate, n);
    launched("ink_mul_sigmoid");
}

void ink_store_ring(const float* x, uint16_t* cache, int n, int64_t pos, int rows, int ring, void* stream) {
    if (rows <= 0) return;
    ink_store_ring_kernel<<<grid((size_t) rows * n, 256), 256, 0, cs(stream)>>>(x, (__half*) cache, n,
                                                                                (long long) pos, rows, ring);
    launched("ink_store_ring");
}

void ink_softcap(float* logits, int n_keep, int n_all, int rows, float cap, void* stream) {
    if (rows <= 0) return;
    ink_softcap_kernel<<<grid((size_t) rows * n_all, 256), 256, 0, cs(stream)>>>(logits, n_keep, n_all, rows, cap);
    launched("ink_softcap");
}

// The warps of a block of the value kernel: as many as have their sums in a block's shared memory.
static int value_warps(int n_head, int n_kv, int dim) {
    return (int) std::min<size_t>(32, kValueShared / ((size_t) (n_head / n_kv) * dim * sizeof(float)));
}

size_t ink_attend_floats(int n_head, int n_kv, int dim, int64_t keys) {
    // a head's scores, its softmax's norm, and its values' sums a block of the value kernel
    const int64_t block = (int64_t) value_warps(n_head, n_kv, dim) * kValueKeys;
    return (size_t) n_head * ((size_t) keys + 1 + (size_t) ((keys + block - 1) / block) * dim);
}

void ink_attend(const InkAttn& a, void* stream) {
    if (a.rows <= 0) return;
    if (a.n_kv < 1 || a.n_head % a.n_kv != 0 || a.n_head / a.n_kv > kMaxPer || (a.dim != 64 && a.dim != 128) ||
        a.n_rel < 1 || a.n_rel > 32)
        throw std::invalid_argument("ink_attend: heads of 64 or 128 values, up to 4 on a key head, up to 32 relative "
                                    "values");
    // the most keys a query of the call reads: with the positions on the card, what the launches are sized for
    const int64_t last = a.at ? a.room : a.pos + a.rows;
    if (a.at && a.room < 1) throw std::invalid_argument("ink_attend: positions on the card need the rows to size for");
    const int nk_max = (int) (a.window > 0 ? std::min<int64_t>(a.window, last) : last);
    const int group = a.n_head / a.n_kv, warps = value_warps(a.n_head, a.n_kv, a.dim);
    const int n_parts = (nk_max + warps * kValueKeys - 1) / (warps * kValueKeys);
    const int score_blocks = (nk_max + kScoreWarps * kScoreKeys - 1) / (kScoreWarps * kScoreKeys);
    const size_t per = ink_attend_floats(a.n_head, a.n_kv, a.dim, nk_max);
    if (a.score_n < per) throw std::logic_error("ink_attend: the scores of one token do not fit the scratch");
    const int chunk = (int) std::min<size_t>(a.score_n / per, (size_t) a.rows);
    float* norm = a.score + (size_t) chunk * a.n_head * nk_max;
    float* parts = norm + (size_t) chunk * a.n_head;
    for (int t0 = 0; t0 < a.rows; t0 += chunk) {
        const int m = std::min(chunk, a.rows - t0);
        launch_pdl(ink_score_kernel, dim3(m * a.n_kv, score_blocks), dim3(kScoreWarps * 32), 0, cs(stream), a, t0,
                   nk_max);
        launched("ink_score");
        launch_pdl(ink_softmax_kernel, dim3(m * a.n_head), dim3(kAttnThreads), 0, cs(stream), a, t0, nk_max, norm);
        launched("ink_softmax");
        launch_pdl(ink_value_kernel, dim3(m * a.n_kv, n_parts), dim3(warps * 32),
                   (size_t) warps * group * a.dim * sizeof(float), cs(stream), a, t0, nk_max, parts, n_parts);
        launched("ink_value");
        launch_pdl(ink_join_kernel, dim3(m * a.n_head), dim3(128), 0, cs(stream), a, t0, parts, n_parts, norm);
        launched("ink_join");
    }
}

void ink_control3_gate(const float* const* cur, const float* const* msg, const float* const* ch,
                       const float* const* gain_w, const float* const* forget_w, float* const* out, int n, int rank,
                       void* stream) {
    if (rank >= 128) throw std::invalid_argument("ink_control3_gate: gemv_set() takes products this wide otherwise");
    Control3 c{};
    for (int j = 0; j < 3; ++j) {
        c.cur[j] = cur[j];
        c.msg[j] = msg[j];
        c.ch[j] = ch[j];
        c.gw[j] = gain_w[j];
        c.fw[j] = forget_w[j];
        c.out[j] = out[j];
    }
    launch_pdl(ink_control3_gate_kernel, dim3((n + 7) / 8), dim3(256), 0, cs(stream), c, n, rank);
    launched("ink_control3_gate");
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream, void* q8,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    if (q8 && n % 32 != 0) throw std::invalid_argument("situ_rows: quantized rows of a multiple of 32");
    launch_pdl(situ_rows_kernel, dim3(grid((size_t) n, 256), rows), dim3(256), 0, cs(stream), gate_up, out, n, beta, lb,
               scales ? *scales : ExpertScales{}, (QB*) q8);
    launched("situ_rows");
}

// ---- the DSpark drafter's kernels (Inkling2::guess_block(), ds_commit()) ----
//
// The drafter's forward (the features' mixes, the context keys and values, the block of the anchor and mask rows, the
// Markov head's recurrence and its fold into the output matrix) follows Basalt's src/models/inkling2/dspark.cpp and
// kernels/dspark.cu (github.com/jesdga95/basalt), which follow vLLM's Inkling 2 DSpark draft (Apache-2.0); the kernels
// here are this engine's own.  Basalt is under the MIT license:
// MIT License
// Copyright (c) 2026 Jesús Gulfo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace {

constexpr int kDsHead = 128;   // a drafter head's values

__device__ __forceinline__ float ds_bf16(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

__device__ __forceinline__ long long ds_pos(const DsPos& p, int t) { return p.at ? p.at[2 * t] : p.pos0 + t; }

// row t (blockIdx.x), mix j (blockIdx.y): v = sum_k alpha[j][k] f_k, then rms(v) * norm[j]; mixes 0..2 to out, mix 3 to
// the ring row of the row's position
__global__ void ds_mix_kernel(DsMix m, int n, int rows, float eps) {
    __shared__ float part[32];
    extern __shared__ float v[];
    const int t = blockIdx.x, j = blockIdx.y;
    const float* a = m.alpha + j * 4;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) x += a[k] * m.tap[(size_t) k * m.tap_stride + (size_t) t * n + i];
        v[i] = x;
        ss += x * x;
    }
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / (float) n + eps);
    float* out = j < 3 ? m.out + ((size_t) j * rows + t) * n
                       : m.ring + (size_t) (ds_pos(m.pos, t) % m.ring_rows) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = v[i] * scale * m.norm[j][i];
}

// row t (blockIdx.x), head h (blockIdx.y), a thread a value: x's head RMS-normed with w, then rotated at the row's
// position (NeoX halves, base theta); into the fp16 cache's cell of the position, or back into x.  With v, v's head
// goes to its cache too.
__global__ void ds_rope_kernel(float* x, int heads, const float* __restrict__ w, float eps, float theta, DsPos p,
                               __half* cache, long long cap, const float* v, __half* vcache) {
    __shared__ float part[kDsHead / 32];
    __shared__ float y[kDsHead];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float* row = x + ((size_t) t * heads + h) * kDsHead;
    const float xi = row[i];
    const float scale = 1.0f / sqrtf(block_sum(xi * xi, part) / (float) kDsHead + eps);
    y[i] = xi * scale * w[i];
    __syncthreads();
    const long long pos = ds_pos(p, t);
    const int half = kDsHead / 2, k = i % half;
    const float freq = powf(theta, -2.0f * (float) k / (float) kDsHead), ang = (float) pos * freq;
    const float c = cosf(ang), s = sinf(ang);
    const float out = i < half ? y[i] * c - y[i + half] * s : y[i - half] * s + y[i] * c;
    if (!cache) {
        row[i] = out;
        return;
    }
    const size_t cell = (size_t) (pos % cap) * heads * kDsHead + (size_t) h * kDsHead + i;
    cache[cell] = __float2half(out);
    if (v) vcache[cell] = __float2half(v[((size_t) t * heads + h) * kDsHead + i]);
}

// query row t (blockIdx.x), head h (blockIdx.y): softmax attention over the cache's positions [lo, hi), every one of
// them for every row (the block's rows see each other)
__global__ void ds_attend_kernel(const float* q, const __half* keys, const __half* vals, long long cap, long long lo,
                                 long long hi, int heads, int kv_heads, float scale, float* score, size_t score_stride,
                                 float* out) {
    __shared__ float qs[kDsHead];
    __shared__ float part[32];
    const int t = blockIdx.x, h = blockIdx.y, g = h / (heads / kv_heads), tid = threadIdx.x;
    const int nk = (int) (hi - lo);
    if (tid < kDsHead) qs[tid] = q[((size_t) t * heads + h) * kDsHead + tid] * scale;
    __syncthreads();
    float* s = score + ((size_t) t * heads + h) * score_stride;
    float m = -INFINITY;
    for (int j = tid; j < nk; j += blockDim.x) {
        const __half* k = keys + (size_t) ((lo + j) % cap) * kv_heads * kDsHead + (size_t) g * kDsHead;
        float d = 0.0f;
        for (int c = 0; c < kDsHead; c += 2) {
            const float2 kv = __half22float2(*(const half2*) (k + c));
            d += qs[c] * kv.x + qs[c + 1] * kv.y;
        }
        s[j] = d;
        m = fmaxf(m, d);
    }
    m = block_max(m, part);
    float sum = 0.0f;
    for (int j = tid; j < nk; j += blockDim.x) {
        const float e = expf(s[j] - m);
        s[j] = e;
        sum += e;
    }
    sum = block_sum(sum, part);
    if (tid >= kDsHead) return;
    float acc = 0.0f;
    for (int j = 0; j < nk; ++j)
        acc += s[j] * __half2float(vals[(size_t) ((lo + j) % cap) * kv_heads * kDsHead + (size_t) g * kDsHead + tid]);
    out[((size_t) t * heads + h) * kDsHead + tid] = acc / sum;
}

__global__ void ds_swiglu_kernel(float* g, const float* __restrict__ u, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

// rows 1 .. rows - 1 of x: the mask row
__global__ void ds_mask_rows_kernel(float* x, const float* __restrict__ mask, int n, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int r = 1; r < rows; ++r) x[(size_t) r * n + i] = mask[i];
}

// [state ; w1's row of the token before] (rank + rank values), the token *prev
__global__ void ds_markov_in_kernel(const float* __restrict__ state, const uint16_t* __restrict__ w1,
                                    const int32_t* __restrict__ prev, float* in, int rank) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rank) return;
    in[r] = state[r];
    in[rank + r] = ds_bf16(w1[(size_t) *prev * rank + r]);
}

// raw = a + b + c + d; gate = sigmoid(raw[0:r]), state = gate * state + (1 - gate) * tanh(raw[r:2r]),
// latent = tanh(raw[2r:3r])
__global__ void ds_markov_step_kernel(const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                      const float* __restrict__ bias, const float* __restrict__ sp, float* state,
                                      float* latent, int rank) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rank) return;
    const auto raw = [&](int k) { return ctx_h[k] + ctx_f[k] + bias[k] + sp[k]; };
    const float gate = 1.0f / (1.0f + expf(-raw(r)));
    state[r] = gate * state[r] + (1.0f - gate) * tanhf(raw(rank + r));
    latent[r] = tanhf(raw(2 * rank + r));
}

// A draft's pick in two launches: each block of the first takes a slice of the vocabulary, its scores
// cap tanh((base + fy + fb) / cap), and leaves its largest (the lowest id of equals) and its softmax sum against it;
// the second's one block joins the slices: the pick, its probability, and the pick as the next step's token.
constexpr int kDsPickBlocks = 128;

struct DsPart {
    float m, s;
    int id;
};

__global__ void ds_pick_part_kernel(const float* __restrict__ base, const float* __restrict__ fy,
                                    const float* __restrict__ fb, int n_keep, float cap, DsPart* part) {
    __shared__ float bm[256 / 32], bs[256 / 32];
    __shared__ int bi[256 / 32];
    const int per = (n_keep + gridDim.x - 1) / gridDim.x, v0 = blockIdx.x * per, v1 = min(n_keep, v0 + per);
    float m = -INFINITY, s = 0.0f;
    int id = 0x7fffffff;
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) {
        const float z = cap * tanhf((base[v] + fy[v] + fb[v]) / cap);
        if (z > m) {
            s = s * expf(m - z) + 1.0f;
            m = z;
            id = v;
        } else {
            s += expf(z - m);
        }
    }
    const auto join = [](float& m, float& s, int& id, float m2, float s2, int id2) {
        if (m2 > m || (m2 == m && id2 < id)) {
            s = (m == -INFINITY ? 0.0f : s * expf(m - m2)) + s2;
            m = m2;
            id = id2;
        } else if (m2 != -INFINITY) {
            s += s2 * expf(m2 - m);
        }
    };
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffff, m, o), s2 = __shfl_xor_sync(0xffffffff, s, o);
        const int id2 = __shfl_xor_sync(0xffffffff, id, o);
        join(m, s, id, m2, s2, id2);
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) bm[w] = m, bs[w] = s, bi[w] = id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < (int) blockDim.x / 32; ++k) join(m, s, id, bm[k], bs[k], bi[k]);
        part[blockIdx.x] = {m, s, id};
    }
}

__global__ void ds_pick_join_kernel(const DsPart* __restrict__ part, int parts, Likeliest* out, int32_t* prev) {
    if (threadIdx.x != 0) return;
    float m = -INFINITY, s = 0.0f;
    int id = 0x7fffffff;
    for (int k = 0; k < parts; ++k) {
        const DsPart p = part[k];
        if (p.m > m || (p.m == m && p.id < id)) {
            s = (m == -INFINITY ? 0.0f : s * expf(m - p.m)) + p.s;
            m = p.m;
            id = p.id;
        } else if (p.m != -INFINITY) {
            s += p.s * expf(p.m - m);
        }
    }
    out->token = id;
    out->prob = 1.0f / s;
    *prev = id;
}

// fold rows' Q8_0 block b (values 32b .. 32b + 31 of each row) from 32 rows of the transposed fold, ft[k][v]
__global__ void ds_fold_q8_kernel(const float* __restrict__ ft, int n_rows, int blocks, int b, uint8_t* fold) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_rows) return;
    float x[32], amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        x[k] = ft[(size_t) k * n_rows + v];
        amax = fmaxf(amax, fabsf(x[k]));
    }
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* blk = fold + ((size_t) v * blocks + b) * 34;
    *(__half*) blk = __float2half(d);
#pragma unroll
    for (int k = 0; k < 32; ++k) blk[2 + k] = (uint8_t) (int8_t) roundf(x[k] * id);
}

}  // namespace

void ds_mix(const DsMix& m, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    ds_mix_kernel<<<dim3(rows, 4), 256, (size_t) n * sizeof(float), cs(stream)>>>(m, n, rows, eps);
    launched("ds_mix");
}

void ds_rope(float* x, int heads, int rows, const float* w, float eps, float theta, const DsPos& p, uint16_t* cache,
             int64_t cap, const float* v, uint16_t* vcache, void* stream) {
    if (rows <= 0) return;
    ds_rope_kernel<<<dim3(rows, heads), kDsHead, 0, cs(stream)>>>(x, heads, w, eps, theta, p, (__half*) cache,
                                                                  (long long) cap, v, (__half*) vcache);
    launched("ds_rope");
}

void ds_attend(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t cap, int64_t lo, int64_t hi,
               int rows, int heads, int kv_heads, float scale, float* score, size_t score_stride, float* out,
               void* stream) {
    ds_attend_kernel<<<dim3(rows, heads), 256, 0, cs(stream)>>>(q, (const __half*) keys, (const __half*) vals,
                                                                (long long) cap, (long long) lo, (long long) hi, heads,
                                                                kv_heads, scale, score, score_stride, out);
    launched("ds_attend");
}

void ds_swiglu(float* g, const float* u, size_t n, void* stream) {
    ds_swiglu_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(g, u, n);
    launched("ds_swiglu");
}

void ds_mask_rows(float* x, const float* mask, int n, int rows, void* stream) {
    ds_mask_rows_kernel<<<grid((size_t) n, 256), 256, 0, cs(stream)>>>(x, mask, n, rows);
    launched("ds_mask_rows");
}

void ds_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, float* in, int rank, void* stream) {
    ds_markov_in_kernel<<<grid((size_t) rank, 256), 256, 0, cs(stream)>>>(state, w1, prev, in, rank);
    launched("ds_markov_in");
}

void ds_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const float* sp, float* state,
                    float* latent, int rank, void* stream) {
    ds_markov_step_kernel<<<grid((size_t) rank, 256), 256, 0, cs(stream)>>>(ctx_h, ctx_f, bias, sp, state, latent,
                                                                           rank);
    launched("ds_markov_step");
}

size_t ds_pick_floats() { return kDsPickBlocks * sizeof(DsPart) / sizeof(float); }

void ds_pick(const float* base, const float* fy, const float* fb, int n_keep, float cap, float* scratch, Likeliest* out,
             int32_t* prev, void* stream) {
    ds_pick_part_kernel<<<kDsPickBlocks, 256, 0, cs(stream)>>>(base, fy, fb, n_keep, cap, (DsPart*) scratch);
    launched("ds_pick_part");
    ds_pick_join_kernel<<<1, 32, 0, cs(stream)>>>((const DsPart*) scratch, kDsPickBlocks, out, prev);
    launched("ds_pick_join");
}

void ds_fold_q8(const float* ft, int n_rows, int blocks, int b, void* fold, void* stream) {
    ds_fold_q8_kernel<<<grid((size_t) n_rows, 256), 256, 0, cs(stream)>>>(ft, n_rows, blocks, b, (uint8_t*) fold);
    launched("ds_fold_q8");
}

}  // namespace strata::poly
