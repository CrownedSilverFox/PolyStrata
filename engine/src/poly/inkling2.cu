// src/poly/inkling2.cu - the kernels of Inkling 2's layers (inkling2_kernels.hpp).  The reference for the math is
// llama.cpp's inkling2 graph (src/models/inkling2.cpp, inkling2-loop.cpp).
//
// A token (or a token's head) a block, sums in a fixed order so that a token comes out the same in any company.
// The attention of a few tokens reads a key or a value with a warp, its lanes side by side: ink_attend().
// q81_div(): a block's scale divided as native_mmvq.cu's --use_fast_math division does it (q8_1_finite.hpp)
#define STRATA_Q81_FASTMATH_DIV 1
#include "inkling2_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }
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

__global__ void ink_mix_kernel(MixArgs a, int n_embd) { mix_row(a, n_embd); }

constexpr int kMixJobs = 12;
struct MixMany {
    MixArgs j[kMixJobs];
};

// blockIdx.y the job
__global__ void ink_mix_many_kernel(MixMany m, int n_embd) { mix_row(m.j[blockIdx.y], n_embd); }

constexpr int kSet = 3;   // the controllers one launch takes

struct CatNormSet {
    const float* cur[kSet];
    const float* msg[kSet];
    float* out[kSet];
};

// A token's pair a block, blockIdx.y the pair.
__global__ void ink_cat_norm_kernel(CatNormSet s, int n_embd) {
    __shared__ float part[256 / WARP];
    const float* __restrict__ cur = s.cur[blockIdx.y];
    const float* __restrict__ msg = s.msg[blockIdx.y];
    float* __restrict__ out = s.out[blockIdx.y];
    const size_t row = size_t(blockIdx.x) * n_embd;
    const int tid = threadIdx.x;
    float a = 0.0f, b = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        a += cur[row + i] * cur[row + i];
        b += msg[row + i] * msg[row + i];
    }
    const float ia = 1.0f / sqrtf(block_sum(a, part) / (float) n_embd + 1e-6f);
    const float ib = 1.0f / sqrtf(block_sum(b, part) / (float) n_embd + 1e-6f);
    float* o = out + 2 * row;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        o[i] = cur[row + i] * ia;
        o[n_embd + i] = msg[row + i] * ib;
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
__device__ __forceinline__ float conv_value(const float* __restrict__ x, const float* __restrict__ state,
                                            const float* __restrict__ w, const float* __restrict__ dyn, int channels,
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
__device__ __forceinline__ void conv_state_at(const float* __restrict__ x, float* state, int channels, int taps,
                                              int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        state[size_t(k) * channels + c] =
            u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
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

__global__ void ink_conv_add_kernel(const float* __restrict__ x, float* state, const float* __restrict__ w,
                                    float* __restrict__ out, float* acc, float* keep, float a, int mode, int channels,
                                    int taps, float norm) {
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

__global__ void ink_moe_conv_add_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ scale, const int* __restrict__ ids, int n_used,
                                        float* fo, float* state, const float* __restrict__ cw, float* __restrict__ out,
                                        float* acc, float a, int channels, int taps, float norm) {
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
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
    for (int j = j0; j < j1; ++j) {
        float k[kLaneHalves];
        lane_halves((const __half*) a.keys + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, k);
        const long long dist = p - (lo + j);
        const float rb = lane < a.n_rel && dist < a.extent ? a.rel_bias[size_t(dist) * a.n_rel + lane] : 0.0f;
#pragma unroll
        for (int hp = 0; hp < kMaxPer; ++hp) {
            if (hp < per) {
                float d = rb * r[hp];
#pragma unroll
                for (int c = 0; c < kLaneHalves; ++c) d += q[hp][c] * k[c];
                d = warp_sum(d);
                if (lane == 0) s[size_t(hp) * nk_max + j] = d - sinks[hp];
            }
        }
        if (++row == a.ring) row = 0;
    }
}

// A token's head a block: the scores' softmax with its extra term of score 0.  The scores become the weights'
// numerators (exp against the largest), and norm gets what they are divided by, a token's head each.
__global__ void ink_softmax_kernel(InkAttn a, int t0, int nk_max, float* norm) {
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
// sums are added in their order.
__global__ void ink_value_kernel(InkAttn a, int t0, int nk_max, float* parts, int n_parts) {
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
    long long row = a.ring > 0 ? (lo + j0) % a.ring : lo + j0;
    for (int j = j0; j < j1; ++j) {
        float v[kLaneHalves];
        lane_halves((const __half*) a.vals + (size_t(row) * a.n_kv + g) * a.dim, lane, lh, v);
#pragma unroll
        for (int hp = 0; hp < kMaxPer; ++hp) {
            if (hp < per) {
                const float w = __shfl_sync(0xffffffff, weight, hp * kValueKeys + (j - j0), WARP);
#pragma unroll
                for (int c = 0; c < kLaneHalves; ++c) acc[hp][c] += w * v[c];
            }
        }
        if (++row == a.ring) row = 0;
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
__global__ void ink_join_kernel(InkAttn a, int t0, const float* __restrict__ parts, int n_parts,
                                const float* __restrict__ norm) {
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
__global__ void situ_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n, float beta,
                                 float lb, ExpertScales s, QB* q8) {
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
    __shared__ float part[32];
    const float* __restrict__ xr = m.x[blockIdx.y];
    const float* __restrict__ w = m.w[blockIdx.y];
    float* yr = m.y[blockIdx.y];
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
__global__ void ink_gate_norm_q8_kernel(const float* cur, const float* __restrict__ msg, const float* __restrict__ gain,
                                        const float* __restrict__ forget, float* out, const float* __restrict__ w,
                                        float* y, QB* q, int n, float eps) {
    __shared__ float part[32];
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
    ink_norm_many_q8_kernel<<<dim3(1, count), sum_threads(n), 0, cs(stream)>>>(m, n, eps);
    launched("ink_norm_many_q8");
}

void ink_gate_norm_q8(const float* cur, const float* msg, const float* gain, const float* forget, float* out,
                      const float* w, float* y, void* q8, int n, float eps, void* stream) {
    const int threads = sum_threads(n);
    if (n % 32 != 0 || threads != std::min(1024, std::max(threads, n)))   // add_rms_norm_q8_1's block is this one
        throw std::invalid_argument("ink_gate_norm_q8: not add_rms_norm_q8_1's block");
    ink_gate_norm_q8_kernel<<<1, threads, 0, cs(stream)>>>(cur, msg, gain, forget, out, w, y, (QB*) q8, n, eps);
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
    ink_mix_kernel<<<rows, kMixThreads, 0, cs(stream)>>>(a, n_embd);
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
    ink_mix_many_kernel<<<dim3(1, jobs), kMixThreads, 0, cs(stream)>>>(m, n_embd);
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
              int taps, int rows, void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv: 2 to 8 taps");
    if (rows <= 0) return;
    const float norm = 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * taps + 1));
    ink_conv_kernel<<<grid((size_t) rows * channels, 256), 256, 0, cs(stream)>>>(x, state, w, dyn, out, channels,
                                                                                  heads, taps, rows, norm);
    launched("ink_conv");
    ink_conv_state_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, channels, taps, rows);
    launched("ink_conv_state");
}

void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    ink_conv_add_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, state, w, out, acc, keep, a, mode,
                                                                            channels, taps, norm);
    launched("ink_conv_add");
}

void ink_moe_conv_add(const float* x, const float* w, const float* scale, const int* ids, int n_used, float* fo,
                      float* state, const float* cw, float* out, float* acc, float a, int channels, int taps,
                      void* stream) {
    if (taps < 2 || taps > 8) throw std::invalid_argument("ink_moe_conv_add: 2 to 8 taps");
    const float norm = 1.0f / std::sqrt((float) (taps + 1));
    ink_moe_conv_add_kernel<<<grid((size_t) channels, 256), 256, 0, cs(stream)>>>(x, w, scale, ids, n_used, fo, state, cw,
                                                                                out, acc, a, channels, taps, norm);
    launched("ink_moe_conv_add");
}

void ink_qkv(const InkQkv& a, void* stream) {
    if (a.dim > 128 || a.n_rel > 128 || a.taps < 2 || a.taps > 8 || a.n_kv < 1)
        throw std::invalid_argument("ink_qkv: heads of up to 128 values, 2 to 8 taps");
    QkvArgs g{a, 1.0f / std::sqrt((float) ((a.kdyn ? 2 : 1) * a.taps + 1)),
              1.0f / std::sqrt((float) ((a.vdyn ? 2 : 1) * a.taps + 1))};
    ink_qkv_kernel<<<2 * a.n_head + 2 * a.n_kv, 128, 0, cs(stream)>>>(g);
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
        ink_score_kernel<<<dim3(m * a.n_kv, score_blocks), kScoreWarps * 32, 0, cs(stream)>>>(a, t0, nk_max);
        launched("ink_score");
        ink_softmax_kernel<<<m * a.n_head, kAttnThreads, 0, cs(stream)>>>(a, t0, nk_max, norm);
        launched("ink_softmax");
        ink_value_kernel<<<dim3(m * a.n_kv, n_parts), warps * 32, (size_t) warps * group * a.dim * sizeof(float),
                           cs(stream)>>>(a, t0, nk_max, parts, n_parts);
        launched("ink_value");
        ink_join_kernel<<<m * a.n_head, 128, 0, cs(stream)>>>(a, t0, parts, n_parts, norm);
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
    ink_control3_gate_kernel<<<(n + 7) / 8, 256, 0, cs(stream)>>>(c, n, rank);
    launched("ink_control3_gate");
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream, void* q8,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    if (q8 && n % 32 != 0) throw std::invalid_argument("situ_rows: quantized rows of a multiple of 32");
    situ_rows_kernel<<<dim3(grid((size_t) n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, beta, lb,
                                                                                scales ? *scales : ExpertScales{},
                                                                                (QB*) q8);
    launched("situ_rows");
}

}  // namespace strata::poly
