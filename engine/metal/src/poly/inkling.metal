// metal/src/poly/inkling.metal - Inkling 2's kernels on Metal (inkling2_kernels.hpp; inkling.cpp launches them).  The
// arithmetic is inkling2.cu's, kernel for kernel: the mixes over the residual streams, the loop controllers' gates,
// the short convolutions, the clips, the attention with its bias by distance and its sink, and SiTU.
#include <metal_stdlib>
using namespace metal;

// Mirrors Args in inkling.cpp; device pointers are Metal gpuAddress values.
struct Args {
    device void* p[72];
    long pos;
    ulong stride;
    int n[24];
    float f[16];
};
template <class T> inline device T* P(constant Args& a, int k) { return static_cast<device T*>(a.p[k]); }

// The threadgroup's sum of v in every thread, the SIMD groups' sums added in their order.
inline float tg_sum(float v, threadgroup float* part, uint tid, uint nt) {
    v = simd_sum(v);
    if ((tid & 31) == 0) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint k = 0; k < nt / 32; ++k) s += part[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---- the mixes over the residual streams (ink_mix(), ink_mix_many(), ink_mix_norm()) ----
// p[0..5] the states, p[6 + job] a job's queries (nq of n), p[18 + 4 job + j] its outputs, p[66] the norm's weight,
// p[67] its output (out[0]'s rms_norm()); n: states, queries, n_embd; f[0] the norm's eps.  A row and a job a
// threadgroup (grid rows x jobs).
kernel void ink_mix(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[30][32];
    threadgroup float coef[6][4];
    threadgroup float npart[32];
    const int ns = a.n[0], nq = a.n[1], n = a.n[2];
    const ulong row = ulong(g.x) * n;
    device const float* q = P<float>(a, 6 + g.y);
    float sum[6][5];
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < 5; ++k) sum[s][k] = 0;
    for (int i = tid; i < n; i += nt) {
        float qv[4];
        for (int j = 0; j < 4; ++j) qv[j] = j < nq ? q[ulong(j) * n + i] : 0.0f;
        for (int s = 0; s < 6; ++s) {
            if (s < ns) {
                const float v = P<float>(a, s)[row + i];
                sum[s][0] += v * v;
                for (int j = 0; j < 4; ++j) sum[s][1 + j] += v * qv[j];
            }
        }
    }
    for (int s = 0; s < 6; ++s)
        for (int k = 0; k < 5; ++k)
            if (s < ns && k <= nq) {
                const float w = simd_sum(sum[s][k]);
                if ((tid & 31) == 0) part[s * 5 + k][tid >> 5] = w;
            }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < nq) {
        const int j = tid;
        const float sc = 1.0f / sqrt(float(n));
        float v[6], inv[6], m = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            float ss = 0, dot = 0;
            for (uint w = 0; w < nt / 32; ++w) {
                ss += part[s * 5][w];
                dot += part[s * 5 + 1 + j][w];
            }
            inv[s] = 1.0f / sqrt(ss / float(n) + 1e-6f);
            v[s] = dot * inv[s] * sc;
            m = max(m, v[s]);
        }
        float total = 0;
        for (int s = 0; s < ns; ++s) {
            v[s] = exp(v[s] - m);
            total += v[s];
        }
        for (int s = 0; s < ns; ++s) coef[s][j] = v[s] / total * inv[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0;
    for (int i = tid; i < n; i += nt) {
        float x[6];
        for (int s = 0; s < 6; ++s) x[s] = s < ns ? P<float>(a, s)[row + i] : 0.0f;
        for (int j = 0; j < nq; ++j) {
            float acc = 0;
            for (int s = 0; s < ns; ++s) acc += coef[s][j] * x[s];
            P<float>(a, 18 + 4 * g.y + j)[row + i] = acc;
            if (j == 0) ss += acc * acc;
        }
    }
    if (!a.p[67]) return;
    const float scale = 1.0f / sqrt(tg_sum(ss, npart, tid, nt) / float(n) + a.f[0]);
    device const float* w = P<float>(a, 66);
    device const float* o = P<float>(a, 18 + 4 * g.y);
    for (int i = tid; i < n; i += nt) P<float>(a, 67)[row + i] = w ? o[row + i] * scale * w[i] : o[row + i] * scale;
}

// ---- the loop controllers ----
// ink_cat_norm_set(): p[j] cur, p[3 + j] msg, p[6 + j] out; n[0] n_embd.  A row and a pair a threadgroup.
kernel void ink_cat_norm(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float part[32];
    const int n = a.n[0];
    const ulong row = ulong(g.x) * n;
    device const float* cur = P<float>(a, g.y) + row;
    device const float* msg = P<float>(a, 3 + g.y) + row;
    device float* out = P<float>(a, 6 + g.y) + 2 * row;
    float sa = 0, sb = 0;
    for (int i = tid; i < n; i += nt) {
        sa += cur[i] * cur[i];
        sb += msg[i] * msg[i];
    }
    const float ia = 1.0f / sqrt(tg_sum(sa, part, tid, nt) / float(n) + 1e-6f);
    const float ib = 1.0f / sqrt(tg_sum(sb, part, tid, nt) / float(n) + 1e-6f);
    for (int i = tid; i < n; i += nt) {
        out[i] = cur[i] * ia;
        out[n + i] = msg[i] * ib;
    }
}

inline float gate_value(float c, float m, float g, float f) {
    return c - precise::tanh(f) * c + precise::tanh(g) * m;
}

// ink_gate_set(): p[j] cur, p[3 + j] msg, p[6 + j] gain, p[9 + j] forget, p[12 + j] out; n[0] the count; stride the
// values.  Every cur of a value is read before an out is written (an out may be a cur).
kernel void ink_gate(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.stride) return;
    const int count = a.n[0];
    float c[3];
    for (int j = 0; j < 3; ++j)
        if (j < count) c[j] = P<float>(a, j)[i];
    for (int j = 0; j < 3; ++j)
        if (j < count)
            P<float>(a, 12 + j)[i] = gate_value(c[j], P<float>(a, 3 + j)[i], P<float>(a, 6 + j)[i], P<float>(a, 9 + j)[i]);
}

inline float wval(constant Args& a, int k, ulong i, bool bf16) {
    return bf16 ? as_type<float>(uint(P<ushort>(a, k)[i]) << 16) : P<float>(a, k)[i];
}

// ink_control3_gate(): p[j] cur, p[3 + j] msg, p[6 + j] ch, p[9 + j] gain, p[12 + j] forget, p[15 + j] out; n: n,
// rank, count, bf16.  A SIMD group a value, 8 a threadgroup.
kernel void ink_control3_gate(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int n = a.n[0], rank = a.n[1], count = a.n[2];
    const bool bf = a.n[3] != 0;
    const int i = int(grp) * 8 + int(sg);
    if (i >= n) return;
    float g[3], f[3];
    for (int j = 0; j < 3; ++j) {
        if (j >= count) break;
        device const float* ch = P<float>(a, 6 + j);
        float ga = 0, fa = 0;
        for (int k = lane; k < rank; k += 32) {
            ga += wval(a, 9 + j, ulong(i) * rank + k, bf) * ch[k];
            fa += wval(a, 12 + j, ulong(i) * rank + k, bf) * ch[k];
        }
        g[j] = simd_sum(ga);
        f[j] = simd_sum(fa);
    }
    if (lane) return;
    float c[3];
    for (int j = 0; j < 3; ++j)
        if (j < count) c[j] = P<float>(a, j)[i];
    for (int j = 0; j < 3; ++j)
        if (j < count) P<float>(a, 15 + j)[i] = gate_value(c[j], P<float>(a, 3 + j)[i], g[j], f[j]);
}

// ---- the short convolutions (ink_conv()) ----
// Token t's convolved value of channel c: past[k] is the input k tokens back, from x or, before the call, the state.
inline float conv_value(device const float* x, device const float* state, device const float* w,
                        device const float* dyn, int channels, int heads, int taps, int t, int c, float norm) {
    float past[8];
    for (int k = 0; k < taps; ++k) {
        const int u = t - k;
        past[k] = u >= 0 ? x[ulong(u) * channels + c] : state[ulong(taps - 1 + u) * channels + c];
    }
    float acc = 0;
    for (int j = 0; j < taps; ++j) acc += past[taps - 1 - j] * w[ulong(c) * taps + j];
    if (dyn) {
        const int head = c / (channels / heads);
        device const float* d = dyn + (ulong(t) * heads + head) * taps;
        for (int lag = 0; lag < taps; ++lag) acc += past[lag] * d[lag];
    }
    return (past[0] + acc) * norm;
}

// Channel c's state after the first `rows` tokens of a call, from the state before it into out (out may be state).
inline void conv_state_into(device const float* x, device const float* state, device float* out, int channels,
                            int taps, int rows, int c) {
    const int keep = taps - 1;
    for (int k = 0; k < keep; ++k) {
        const int u = rows - keep + k;
        out[ulong(k) * channels + c] = u >= 0 ? x[ulong(u) * channels + c] : state[ulong(keep + u) * channels + c];
    }
}

// The snapshots of a window (bit t of bits: the state after token t, the j-th at snap + j * stride), then the state.
inline void conv_states(device const float* x, device float* state, device float* snap, ulong stride, uint bits,
                        int channels, int taps, int rows, int c) {
    int k = 0;
    if (snap)
        for (uint b = bits; b; b &= b - 1, ++k)
            conv_state_into(x, state, snap + ulong(k) * stride, channels, taps, int(ctz(b)) + 1, c);
    conv_state_into(x, state, state, channels, taps, rows, c);
}

// ink_conv_few(): one or two jobs (grid y), a channel a thread, a call of a few tokens.  Job j: p[8j ..] x, state, w,
// dyn, out, snap, acc, keep; n: channels, heads, taps, rows, mode0, mode1, bits; f: a0, a1, norm0, norm1; stride.
kernel void ink_conv_few(constant Args& a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int channels = a.n[0], heads = a.n[1], taps = a.n[2], rows = a.n[3], j = g.y, c = g.x;
    if (c >= channels) return;
    const int b = 8 * j, mode = a.n[4 + j];
    const float s = a.f[j], norm = a.f[2 + j];
    device const float* x = P<float>(a, b);
    device float* state = P<float>(a, b + 1);
    device float* out = P<float>(a, b + 4);
    device float* acc = P<float>(a, b + 6);
    device float* keep = P<float>(a, b + 7);
    for (int t = 0; t < rows; ++t) {
        const ulong i = ulong(t) * channels + c;
        const float y = conv_value(x, state, P<float>(a, b + 2), P<float>(a, b + 3), channels, heads, taps, t, c, norm);
        out[i] = y;
        if (!acc) continue;
        if (mode == 0) {
            acc[i] = acc[i] + y * s;
        } else {
            if (mode == 1) keep[i] = acc[i];
            acc[i] = y;
        }
    }
    conv_states(x, state, P<float>(a, b + 5), a.stride, uint(a.n[6]), channels, taps, rows, c);
}

// ink_conv() of many tokens: p x, state, w, dyn, out; n: channels, heads, taps, rows; f[0] norm.  A value a thread;
// ink_conv_tail then the snapshots and the state.
kernel void ink_conv_rows(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const int channels = a.n[0];
    if (i >= uint(channels) * uint(a.n[3])) return;
    P<float>(a, 4)[i] = conv_value(P<float>(a, 0), P<float>(a, 1), P<float>(a, 2), P<float>(a, 3), channels, a.n[1],
                                   a.n[2], int(i / channels), int(i % channels), a.f[0]);
}

// p x, state, snap; n: channels, taps, rows, bits; stride.  A channel a thread.
kernel void ink_conv_tail(constant Args& a [[buffer(0)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(a.n[0])) return;
    conv_states(P<float>(a, 0), P<float>(a, 1), P<float>(a, 2), a.stride, uint(a.n[3]), a.n[0], a.n[1], a.n[2], c);
}

// ink_moe_conv_add(): p x, w, scale, ids, fo, state, cw, out, acc; n: n_used, channels, taps; f: a, norm.
kernel void ink_moe_conv_add(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    const int used = a.n[0], channels = a.n[1], taps = a.n[2];
    if (i >= uint(channels)) return;
    device const float* x = P<float>(a, 0);
    device const float* w = P<float>(a, 1);
    device const float* scale = P<float>(a, 2);
    device const int* ids = P<int>(a, 3);
    float s = 0;
    for (int k = 0; k < used; ++k) {
        const float wk = scale ? w[k] * scale[ids[k]] : w[k];
        s += wk * x[ulong(k) * channels + i];
    }
    device float* fo = P<float>(a, 4);
    device float* state = P<float>(a, 5);
    fo[i] = 0.0f + s;
    const float y = conv_value(fo, state, P<float>(a, 6), nullptr, channels, 1, taps, 0, int(i), a.f[1]);
    P<float>(a, 7)[i] = y;
    conv_state_into(fo, state, state, channels, taps, 1, int(i));
    device float* acc = P<float>(a, 8);
    acc[i] = acc[i] + y * a.f[0];
}

// ---- the clips and head norms ----
// One head's clip (ink_clip()): the threadgroup's threads over its values; p the token's position.
inline void clip_head(device float* v, device const float* sc, float thr, int dim, float alpha, float floor_, long p,
                      threadgroup float* part, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < dim; i += nt) {
        const float y = sc ? v[i] * sc[i] : v[i];
        ss += y * y;
    }
    float f = min(thr / (sqrt(tg_sum(ss, part, tid, nt)) + 1e-6f), 1.0f);
    if (alpha > 0.0f) f *= 1.0f + alpha * log(max(1.0f, float(p + 1) / floor_));
    for (int i = tid; i < dim; i += nt) v[i] = (sc ? v[i] * sc[i] : v[i]) * f;
}

inline void rms_head(device float* v, int dim, float eps, threadgroup float* part, uint tid, uint nt) {
    float ss = 0;
    for (int i = tid; i < dim; i += nt) ss += v[i] * v[i];
    const float f = 1.0f / sqrt(tg_sum(ss, part, tid, nt) / float(dim) + eps);
    for (int i = tid; i < dim; i += nt) v[i] *= f;
}

inline void store_head(device const float* v, device half* cache, long cell, int heads, int dim, int h, uint tid,
                       uint nt) {
    device half* to = cache + ulong(cell) * heads * dim + ulong(h) * dim;
    for (int i = tid; i < dim; i += nt) to[i] = half(v[i]);
}

// ink_clips(): job j's x p[3j], scale p[3j + 1], cache p[3j + 2]; p[12] at; n: dim[4], heads[4], rms[4], rows, ring;
// f: thr[4], alpha[4], floor[4], eps; pos.  A token's head of a job a threadgroup of 128.
kernel void ink_clips(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const int rows = a.n[12];
    int b = int(grp), j = 0;
    while (j < 3 && b >= rows * a.n[4 + j]) b -= rows * a.n[4 + j++];
    const int heads = a.n[4 + j], dim = a.n[j], row = b / heads, head = b % heads;
    device const long* at = P<long>(a, 12);
    device float* v = P<float>(a, 3 * j) + (ulong(row) * heads + head) * dim;
    if (a.n[8 + j]) {
        rms_head(v, dim, a.f[12], part, tid, nt);
    } else {
        device const float* sc = P<float>(a, 3 * j + 1);
        clip_head(v, sc ? sc + ulong(head) * dim : nullptr, a.f[j], dim, a.f[4 + j], a.f[8 + j],
                  at ? at[2 * row] : a.pos + row, part, tid, nt);
    }
    if (!a.p[3 * j + 2]) return;
    threadgroup_barrier(mem_flags::mem_device);
    long cell = (at ? at[0] : a.pos) + row;
    if (a.n[13] > 0) cell %= a.n[13];
    store_head(v, P<half>(a, 3 * j + 2), cell, heads, dim, head, tid, nt);
}

// ink_qkv(): one token.  p: q, q_scale, r, r_scale, kraw, vraw, kstate, vstate, kw, vw, kdyn, vdyn, k, v, keys, vals,
// at; n: n_head, n_kv, dim, n_rel, taps, ring; f: qk_thr, rel_thr, alpha, floor, eps, k_norm, v_norm; pos.
// Threadgroups [0, n_head) Q's heads, then K's n_kv, V's n_kv, and the relative positions' n_head; 128 threads.
kernel void ink_qkv(constant Args& a [[buffer(0)]], uint grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const int n_head = a.n[0], n_kv = a.n[1], dim = a.n[2], n_rel = a.n[3], taps = a.n[4], ring = a.n[5];
    const long p = a.p[16] ? P<long>(a, 16)[0] : a.pos;
    int b = int(grp);
    if (b < n_head) {
        device const float* sc = P<float>(a, 1);
        clip_head(P<float>(a, 0) + ulong(b) * dim, sc ? sc + ulong(b) * dim : nullptr, a.f[0], dim, a.f[2], a.f[3], p,
                  part, tid, nt);
        return;
    }
    b -= n_head;
    if (b < 2 * n_kv) {
        const bool is_v = b >= n_kv;
        const int h = is_v ? b - n_kv : b, channels = n_kv * dim;
        device const float* x = P<float>(a, is_v ? 5 : 4);
        device float* state = P<float>(a, is_v ? 7 : 6);
        device float* out = P<float>(a, is_v ? 13 : 12);
        if (int(tid) < dim) {
            const int c = h * dim + tid;
            out[c] = conv_value(x, state, P<float>(a, is_v ? 9 : 8), P<float>(a, is_v ? 11 : 10), channels, n_kv, taps,
                                0, c, a.f[is_v ? 6 : 5]);
            conv_state_into(x, state, state, channels, taps, 1, c);
        }
        threadgroup_barrier(mem_flags::mem_device);
        device float* head = out + ulong(h) * dim;
        if (is_v) rms_head(head, dim, a.f[4], part, tid, nt);
        else clip_head(head, nullptr, a.f[0], dim, 0.0f, 1.0f, p, part, tid, nt);
        threadgroup_barrier(mem_flags::mem_device);
        store_head(head, P<half>(a, is_v ? 15 : 14), ring > 0 ? p % ring : p, n_kv, dim, h, tid, nt);
        return;
    }
    b -= 2 * n_kv;
    device const float* rs = P<float>(a, 3);
    clip_head(P<float>(a, 2) + ulong(b) * n_rel, rs ? rs + ulong(b) * n_rel : nullptr, a.f[1], n_rel, a.f[2], a.f[3],
              p, part, tid, nt);
}

// ---- elementwise: n[0] the kind, stride the count ----
// 0: x *= sigmoid(gate) (p x, gate); 1: an fp16 ring's rows (p x, cache; n[1] n, n[2] ring; pos); 2: the soft cap
// (p logits; n[1] n_keep, n[2] n_all; f[0] cap)
kernel void ink_elem(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.stride) return;
    switch (a.n[0]) {
        case 0: P<float>(a, 0)[i] *= sigm(P<float>(a, 1)[i]); break;
        case 1: {
            const int n = a.n[1];
            P<half>(a, 1)[ulong((a.pos + long(i / n)) % a.n[2]) * n + i % n] = half(P<float>(a, 0)[i]);
            break;
        }
        case 2: {
            device float* l = P<float>(a, 0);
            l[i] = int(i % uint(a.n[2])) < a.n[1] ? a.f[0] * precise::tanh(l[i] / a.f[0]) : -INFINITY;
            break;
        }
    }
}

// ---- SiTU (situ_rows()) ----
struct SituArgs {
    device void* p[5];   // gate_up, out, gate scale, up scale, ids
    int n, rows;
    float beta, lb;
    int e[32];
};
kernel void ink_situ(constant SituArgs& a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int i = g.x, r = g.y;
    if (i >= a.n) return;
    device const float* row = static_cast<device float*>(a.p[0]) + ulong(r) * 2 * a.n;
    float sg = 1, su = 1;
    if (a.p[2] || a.p[3]) {
        const int e = a.p[4] ? static_cast<device int*>(a.p[4])[r] : a.e[r];
        if (a.p[2]) sg = static_cast<device float*>(a.p[2])[e];
        if (a.p[3]) su = static_cast<device float*>(a.p[3])[e];
    }
    const float gv = row[i] * sg;
    float u = row[a.n + i] * su;
    if (a.lb > 0.0f) u = a.lb * precise::tanh(u / a.lb);
    static_cast<device float*>(a.p[1])[ulong(r) * a.n + i] = a.beta * precise::tanh(gv / a.beta) * sigm(gv) * u;
}

// ---- the attention (ink_attend()) ----
// p: q, r, sink, keys, vals, rel_bias, gate, out, at, runs; n: rows, n_head, n_kv, dim, n_rel, extent, window, ring,
// splits; f[0] the scale; pos.  A threadgroup a key head, a token and a run of its keys (grid n_kv x rows x splits):
// its SIMD groups take every g-th key, each with a softmax of its own for each of the key head's query heads, lanes
// side by side on a key (PER values a lane).  A key at distance d scores scale * q.k + r . rel_bias[d] (0 from
// `extent` on) - sink.  The SIMD groups meet in threadgroup memory; with splits the runs go to `runs` for
// ink_attn_join, else the softmax's extra term of score 0 and the gate are taken here.
inline long ink_pos(constant Args& a, int t) { return a.p[8] ? P<long>(a, 8)[2 * t] : a.pos + t; }

template <int PER>
kernel void ink_attn(constant Args& a [[buffer(0)]], threadgroup float* sh [[threadgroup(0)]],
                     uint3 grp [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
    const int g = grp.x, t = grp.y, z = grp.z;
    const int nh = a.n[1], nkv = a.n[2], dim = a.n[3], nrel = a.n[4], extent = a.n[5], window = a.n[6],
              ring = a.n[7], splits = a.n[8];
    const int per = nh / nkv;
    const long p = ink_pos(a, t);
    long lo = window > 0 ? max(0L, p - window + 1) : 0, hi = p;
    if (splits > 1) {
        const long span = (hi - lo + splits) / splits, first = lo + long(z) * span;
        hi = min(hi, first + span - 1);
        lo = first;
    }
    const ulong head0 = ulong(t) * nh + ulong(g) * per;
    float q[4][PER], r[4], sink[4];
    for (int hp = 0; hp < 4; ++hp) {
        for (int c = 0; c < PER; ++c) q[hp][c] = hp < per ? P<float>(a, 0)[(head0 + hp) * dim + lane * PER + c] * a.f[0] : 0;
        r[hp] = hp < per && int(lane) < nrel ? P<float>(a, 1)[(head0 + hp) * nrel + lane] : 0;
        sink[hp] = hp < per ? P<float>(a, 2)[head0 + hp] : 0;
    }
    float m[4], l[4], acc[4][PER];
    for (int hp = 0; hp < 4; ++hp) {
        m[hp] = -INFINITY;
        l[hp] = 0;
        for (int c = 0; c < PER; ++c) acc[hp][c] = 0;
    }
    device const half* keys = P<half>(a, 3);
    device const half* vals = P<half>(a, 4);
    device const float* rel = P<float>(a, 5);
    for (long j = lo + long(sg); j <= hi; j += long(groups)) {
        const ulong at = (ulong(ring > 0 ? j % ring : j) * nkv + g) * dim + lane * PER;
        float k[PER], v[PER];
        for (int c = 0; c < PER; ++c) {
            k[c] = float(keys[at + c]);
            v[c] = float(vals[at + c]);
        }
        const long dist = p - j;
        const float rb = int(lane) < nrel && dist < extent ? rel[ulong(dist) * nrel + lane] : 0.0f;
        for (int hp = 0; hp < 4; ++hp) {
            if (hp >= per) break;
            float d = rb * r[hp];
            for (int c = 0; c < PER; ++c) d += q[hp][c] * k[c];
            d = simd_sum(d) - sink[hp];
            const float next = max(m[hp], d), keep = exp(m[hp] - next), w = exp(d - next);
            l[hp] = l[hp] * keep + w;
            m[hp] = next;
            for (int c = 0; c < PER; ++c) acc[hp][c] = acc[hp][c] * keep + w * v[c];
        }
    }
    // each SIMD group's per heads: their maxima, their sums, then their dim sums
    const int stride = 2 + dim;
    for (int hp = 0; hp < per; ++hp) {
        threadgroup float* o = sh + (ulong(sg) * per + hp) * stride;
        if (lane == 0) {
            o[0] = m[hp];
            o[1] = l[hp];
        }
        for (int c = 0; c < PER; ++c) o[2 + lane * PER + c] = acc[hp][c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg >= uint(per)) return;
    const int hp = sg;   // SIMD group hp joins head hp's
    float top = -INFINITY;
    for (uint s = 0; s < groups; ++s) top = max(top, sh[(ulong(s) * per + hp) * stride]);
    if (splits <= 1) top = max(top, 0.0f);   // the extra term of score 0
    float total = 0;
    for (uint s = 0; s < groups; ++s) {
        threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
        total += o[1] * exp(o[0] - top);
    }
    const ulong head = head0 + hp;
    if (splits > 1) {
        device float* run = P<float>(a, 9) + (head * splits + z) * stride;
        if (lane == 0) {
            run[0] = top;
            run[1] = total;
        }
        for (int d = lane; d < dim; d += 32) {
            float sum = 0;
            for (uint s = 0; s < groups; ++s) {
                threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
                sum += o[2 + d] * exp(o[0] - top);
            }
            run[2 + d] = sum;
        }
        return;
    }
    total += exp(-top);
    device const float* gate = P<float>(a, 6);
    for (int d = lane; d < dim; d += 32) {
        float sum = 0;
        for (uint s = 0; s < groups; ++s) {
            threadgroup float* o = sh + (ulong(s) * per + hp) * stride;
            sum += o[2 + d] * exp(o[0] - top);
        }
        float y = sum / total;
        if (gate) y *= sigm(gate[head * dim + d]);
        P<float>(a, 7)[head * dim + d] = y;
    }
}
typedef decltype(ink_attn<4>) ink_attn_t;
template [[host_name("ink_attn_128")]] kernel ink_attn_t ink_attn<4>;
template [[host_name("ink_attn_64")]] kernel ink_attn_t ink_attn<2>;

// The splits' runs of a token's head into its output, with the extra term of score 0 and the gate.  A SIMD group a
// token's head (grid n_head x rows).
kernel void ink_attn_join(constant Args& a [[buffer(0)]], uint2 grp [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const int nh = a.n[1], dim = a.n[3], splits = a.n[8], stride = 2 + dim;
    const ulong head = ulong(grp.y) * nh + grp.x;
    device const float* run = P<float>(a, 9) + head * splits * stride;
    float top = 0;   // the extra term's score
    for (int s = 0; s < splits; ++s) top = max(top, run[s * stride]);
    float total = exp(-top);
    for (int s = 0; s < splits; ++s)
        if (run[s * stride + 1] > 0) total += run[s * stride + 1] * exp(run[s * stride] - top);
    device const float* gate = P<float>(a, 6);
    for (int d = lane; d < dim; d += 32) {
        float sum = 0;
        for (int s = 0; s < splits; ++s)
            if (run[s * stride + 1] > 0) sum += run[s * stride + 2 + d] * exp(run[s * stride] - top);
        float y = sum / total;
        if (gate) y *= sigm(gate[head * dim + d]);
        P<float>(a, 7)[head * dim + d] = y;
    }
}
