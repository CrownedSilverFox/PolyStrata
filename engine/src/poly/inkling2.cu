// src/poly/inkling2.cu - the kernels of Inkling 2's layers (inkling2_kernels.hpp).  The reference for the math is
// llama.cpp's inkling2 graph (src/models/inkling2.cpp, inkling2-loop.cpp).
//
// A token (or a token's head) a block, sums in a fixed order so that a token comes out the same in any company.
// The attention of a few tokens reads a key or a value with a warp, its lanes side by side: ink_attend().
#include "inkling2_kernels.hpp"

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
};

// A token a block.  A thread has every kMixThreads-th value of the row and reads each query's and each state's
// once: the states' squares and their products with the queries are summed over the warp, then over the warps in
// their order.
__global__ void ink_mix_kernel(MixArgs a, int n_embd) {
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
}

__global__ void ink_cat_norm_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                    float* __restrict__ out, int n_embd) {
    __shared__ float part[256 / WARP];
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

__global__ void ink_gate_kernel(const float* cur, const float* __restrict__ msg, const float* __restrict__ gain,
                                const float* __restrict__ forget, float* out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = cur[i];
    out[i] = c - tanhf(forget[i]) * c + tanhf(gain[i]) * msg[i];
}

__global__ void ink_conv_kernel(const float* __restrict__ x, const float* __restrict__ state,
                                const float* __restrict__ w, const float* __restrict__ dyn, float* __restrict__ out,
                                int channels, int heads, int taps, int rows, float norm) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(rows) * channels) return;
    const int t = (int) (idx / channels), c = (int) (idx % channels);
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
    out[idx] = (past[0] + acc) * norm;
}

// a channel a thread: the last taps - 1 inputs, oldest first
__global__ void ink_conv_state_kernel(const float* __restrict__ x, float* state, int channels, int taps, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        state[size_t(k) * channels + c] =
            u >= 0 ? x[size_t(u) * channels + c] : state[size_t(keep + u) * channels + c];
    }
}

// a token's head a block
__global__ void ink_clip_kernel(float* x, const float* __restrict__ scale, float thr, int dim, int heads,
                                long long pos, float alpha, float floor_, const long long* __restrict__ at) {
    __shared__ float part[128 / WARP];
    const int row = blockIdx.x / heads, head = blockIdx.x % heads, tid = threadIdx.x;
    float* v = x + (size_t(row) * heads + head) * dim;
    const float* sc = scale ? scale + size_t(head) * dim : nullptr;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    const float tot = block_sum(ss, part);
    float f = fminf(thr / (sqrtf(tot) + 1e-6f), 1.0f);
    if (alpha > 0.0f) f *= 1.0f + alpha * logf(fmaxf(1.0f, (float) ((at ? at[2 * row] : pos + row) + 1) / floor_));
    for (int i = tid; i < dim; i += blockDim.x) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

__global__ void ink_head_rms_kernel(float* x, int dim, int heads, float eps) {
    __shared__ float part[128 / WARP];
    const int tid = threadIdx.x;
    float* v = x + size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) ss += v[i] * v[i];
    const float f = 1.0f / sqrtf(block_sum(ss, part) / (float) dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) v[i] *= f;
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
        a.out[(size_t(t) * a.n_head + h) * a.dim + i] = total * inv;
    }
}

__device__ __forceinline__ float2 situ_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

__global__ void situ_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n, float beta,
                                 float lb, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = situ_scales(s, blockIdx.y);
    const float g = row[i] * sc.x;
    float u = row[n + i] * sc.y;
    if (lb > 0.0f) u = lb * tanhf(u / lb);
    out[size_t(blockIdx.y) * n + i] = beta * tanhf(g / beta) / (1.0f + expf(-g)) * u;
}

}  // namespace

void ink_mix(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
             void* stream) {
    if (states.n < 1 || states.n > InkStates::kMax || n_query < 1 || n_query > kQueries)
        throw std::invalid_argument("ink_mix: 1 to 6 states and 1 to 4 queries");
    if (rows <= 0) return;
    MixArgs a{};
    for (int s = 0; s < states.n; ++s) a.s[s] = states.p[s];
    a.n = states.n;
    a.query = query;
    a.nq = n_query;
    for (int j = 0; j < n_query; ++j) a.out[j] = out[j];
    ink_mix_kernel<<<rows, kMixThreads, 0, cs(stream)>>>(a, n_embd);
    launched("ink_mix");
}

void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream) {
    if (rows <= 0) return;
    ink_cat_norm_kernel<<<rows, 256, 0, cs(stream)>>>(cur, msg, out, n_embd);
    launched("ink_cat_norm");
}

void ink_gate(const float* cur, const float* msg, const float* gain, const float* forget, float* out, size_t n,
              void* stream) {
    if (n == 0) return;
    ink_gate_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(cur, msg, gain, forget, out, n);
    launched("ink_gate");
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

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    situ_rows_kernel<<<dim3(grid((size_t) n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, beta, lb,
                                                                                scales ? *scales : ExpertScales{});
    launched("situ_rows");
}

}  // namespace strata::poly
