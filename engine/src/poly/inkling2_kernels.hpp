// src/poly/inkling2_kernels.hpp - the kernels only Inkling 2's layers use (inkling2.cpp).  CUDA and HIP.
#pragma once

#include "strata/poly/kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// The residual streams a mix reads: the earlier blocks' inputs and the current stream, a row a token each.
struct InkStates {
    static constexpr int kMax = 6;
    const float* p[kMax] = {};
    int n = 0;
};

/// The model's attention over the depth: each of a token's n states is RMS-normed (eps 1e-6, no weight), scored
/// against a learned query (their dot product over sqrt(n_embd)), and the softmax of the scores weighs the normed
/// states' sum.  `query` holds n_query queries of n_embd; out[j] gets query j's rows.
void ink_mix(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
             void* stream);
/// ink_mix() and then rms_norm() of out[0] (weight norm_w, null: none) into norm_out, in one launch with the same bits.
void ink_mix_norm(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
                  const float* norm_w, float* norm_out, float eps, void* stream);

/// out = [rms_norm(cur) | rms_norm(msg)] a token (eps 1e-6, no weight): what a loop controller's down matrix takes.
void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream);
/// ink_cat_norm() of 1 to 3 pairs in one launch.
void ink_cat_norm_set(const float* const* cur, const float* const* msg, float* const* out, int count, int n_embd,
                      int rows, void* stream);

/// ink_gate() of 1 to 3 controllers in one launch.  Every cur is read before any out is written, so an out may be
/// any of the cur.
void ink_gate_set(const float* const* cur, const float* const* msg, const float* const* gain,
                  const float* const* forget, float* const* out, int count, size_t n, void* stream);

/// out = cur - tanh(forget) * cur + tanh(gain) * msg, n values.  out may be cur.
void ink_gate(const float* cur, const float* msg, const float* gain, const float* forget, float* out, size_t n,
              void* stream);

/// The causal short convolution with its residual, for `rows` tokens of `channels` values:
/// out = (x[t] + sum_j x[t - taps + 1 + j] * w[j] + sum_lag x[t - lag] * dyn[t][lag + taps * head]) * norm,
/// norm = 1 / sqrt((dyn ? 2 : 1) * taps + 1).  w is [taps] a channel; dyn (null: none) is taps * heads a token, a
/// channel's head being its place in channels / heads.  `state` holds the taps - 1 inputs before the first token,
/// oldest first, a row of channels each, and is left holding the last taps - 1 of this call's.  out is not x.
void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels, int heads,
              int taps, int rows, void* stream);

/// One token's ink_conv() (no dyn, one head) and its state, out written, then the stream acc updated: mode 0
/// acc = acc + out * a (axpy()), mode 1 keep = acc and then acc = out, mode 2 acc = out.  The same bits.
void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream);
/// One token's routed experts' sum onto fo (moe_sum(): fo += the n_used rows of x weighted by w), then ink_conv_add()
/// of fo in mode 0, in one launch with the same bits.
void ink_moe_conv_add(const float* x, const float* w, int n_used, float* fo, float* state, const float* cw, float* out,
                      float* acc, float a, int channels, int taps, void* stream);

/// Each head's `dim` values, times scale (null: none; [dim] a head), brought down to an L2 norm of at most `thr`
/// (the norm has 1e-6 added), then times tau = 1 + alpha * log(max(1, (pos + row + 1) / floor)) when alpha > 0.
/// With `at` (a recorded launch), row t's position is what the card holds at at[2 * t] when the launch runs.
void ink_clip(float* x, const float* scale, float thr, int dim, int heads, int rows, int64_t pos, float alpha,
              float floor, void* stream, const long long* at = nullptr);

/// Each head's `dim` values divided by their RMS (eps inside the root, no weight).
void ink_head_rms(float* x, int dim, int heads, int rows, float eps, void* stream);

/// x *= sigmoid(gate), n values.
void ink_mul_sigmoid(float* x, const float* gate, size_t n, void* stream);

/// `rows` tokens' rows of n values into an fp16 cache as rows (pos + t) % ring.
void ink_store_ring(const float* x, uint16_t* cache, int n, int64_t pos, int rows, int ring, void* stream);

/// logits[i] = cap * tanh(logits[i] / cap) for the first n_keep, -infinity for the rest of n_all, a row each.
void ink_softcap(float* logits, int n_keep, int n_all, int rows, float cap, void* stream);

/// One token's ink_conv() of K and V with their states, ink_clip() of Q, K and the relative positions' values,
/// ink_head_rms() of V, and the stores of K and V as cache row p, or p % ring when ring > 0 (p: pos, or at[0] when the
/// card holds it), in one launch with the same bits.  Heads of up to 128 values.
struct InkQkv {
    float* q = nullptr;                 ///< n_head * dim, clipped in place
    const float* q_scale = nullptr;
    float* r = nullptr;                 ///< n_head * n_rel, clipped in place
    const float* r_scale = nullptr;
    const float* kraw = nullptr;        ///< n_kv * dim: the products the convolutions take
    const float* vraw = nullptr;
    float* kstate = nullptr;
    float* vstate = nullptr;
    const float* kw = nullptr;
    const float* vw = nullptr;
    const float* kdyn = nullptr;        ///< null: none
    const float* vdyn = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    uint16_t* keys = nullptr;
    uint16_t* vals = nullptr;
    const long long* at = nullptr;
    int64_t pos = 0;
    float qk_thr = 0.f, rel_thr = 0.f, alpha = 0.f, floor = 1.f, eps = 0.f;
    int n_head = 0, n_kv = 0, dim = 0, n_rel = 0, taps = 0, ring = 0;
};
void ink_qkv(const InkQkv& a, void* stream);

/// The attention of `rows` tokens from position pos over the keys and values of an fp16 cache.  A key at distance
/// d = the query's position - its own scores scale * q.k + bias(d) - sink, where bias(d) is the dot product of the
/// query's n_rel values (r) with rel_bias's column d, and 0 from `extent` on; a softmax with one more term of
/// score 0 weighs the values.  With `window` > 0 only the keys at distances under it are read.  The cache's row
/// of a position is the position, or its remainder by `ring` when ring > 0.  Head h reads the keys of head
/// h / (n_head / n_kv).  `score` is scratch of score_n floats, ink_attend_floats() a token: the tokens are taken
/// as many at a time as fit.  Heads of 64 or 128 values, up to 4 on a key head, up to 32 relative values.
struct InkAttn {
    const float* q = nullptr;        ///< rows * n_head * dim
    const float* r = nullptr;        ///< rows * n_head * n_rel
    const float* sink = nullptr;     ///< rows * n_head
    const uint16_t* keys = nullptr;  ///< n_kv * dim halves a cache row
    const uint16_t* vals = nullptr;
    const float* rel_bias = nullptr; ///< n_rel values a distance, `extent` distances
    float* out = nullptr;            ///< rows * n_head * dim
    float* score = nullptr;
    size_t score_n = 0;
    int64_t pos = 0;
    /// a recorded launch: token t's position is what the card holds at at[2 * t] when the launch runs, and `room`
    /// is the most rows such a token reads (what the scratch is laid out for)
    const long long* at = nullptr;
    int room = 0;
    int rows = 0, n_head = 0, n_kv = 0, dim = 0, n_rel = 0, extent = 0, window = 0, ring = 0;
    float scale = 1.f;
};
void ink_attend(const InkAttn& a, void* stream);

/// The floats of scratch ink_attend() needs for a token that reads up to `keys` keys.
size_t ink_attend_floats(int n_head, int n_kv, int dim, int64_t keys);

}  // namespace strata::poly
