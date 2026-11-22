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
/// ink_mix() of one row for `jobs` queries sets (up to 12) over the same states, in one launch with the same bits:
/// job k's n_query queries at query[k], its outputs out[k * n_query + j].
void ink_mix_many(const InkStates& states, const float* const* query, int n_query, float* const* out, int jobs,
                  int n_embd, void* stream);
/// ink_mix() and then rms_norm() of out[0] (weight norm_w, null: none) into norm_out, in one launch with the same bits.
void ink_mix_norm(const InkStates& states, const float* query, int n_query, float* const* out, int n_embd, int rows,
                  const float* norm_w, float* norm_out, float eps, void* stream, void* q8 = nullptr, bool ffn = false);
/// With q8 (one row): the norm also quantized there as native_quantize_q8_1() does it; with ffn the norm is
/// native_add_rms_norm_q8_1()'s instead of rms_norm()'s (and q8 is required).  The same bytes.

/// One row's rms_norm_many() of 1 to 3 inputs (a rows a launch of sum_threads(n) threads), each also quantized into
/// q8[j] as native_quantize_q8_1() does it.  The same bytes.
void ink_norm_many_q8(const float* const* x, const float* const* w, float* const* y, void* const* q8, int count, int n,
                      float eps, void* stream);

/// One row's ink_gate() into out, then native_add_rms_norm_q8_1() of out (weight w) into y and q8.  The same bytes.
void ink_gate_norm_q8(const float* cur, const float* msg, const float* gain, const float* forget, float* out,
                      const float* w, float* y, void* q8, int n, float eps, void* stream);

/// out = [rms_norm(cur) | rms_norm(msg)] a token (eps 1e-6, no weight): what a loop controller's down matrix takes.
void ink_cat_norm(const float* cur, const float* msg, float* out, int n_embd, int rows, void* stream);
/// ink_cat_norm() of 1 to 3 pairs in one launch.
void ink_cat_norm_set(const float* const* cur, const float* const* msg, float* const* out, int count, int n_embd,
                      int rows, void* stream);
/// ink_cat_norm_set() and then gemv_set() of each pair with its controller's F32 down matrix (rank outputs of
/// 2 * n_embd values) into ch[j], in one launch with the same bits.  False, with nothing launched, where gemv_set()
/// would run another block.
bool ink_ctrl_down(const float* const* cur, const float* const* msg, const float* const* down, float* const* ch,
                   int count, int n_embd, int rank, int rows, void* stream);

/// ink_gate() of 1 to 3 controllers in one launch.  Every cur is read before any out is written, so an out may be
/// any of the cur.
/// One token's three controllers after their down products: value i's gain and forget (gemv_set()'s, rank values
/// wide, rank under 128) and its gate (ink_gate()), a warp a value for the three, in one launch with the same bits.
/// Every cur is read before an out is written.
void ink_control3_gate(const float* const* cur, const float* const* msg, const float* const* ch,
                       const float* const* gain_w, const float* const* forget_w, float* const* out, int n, int rank,
                       void* stream);
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
/// With snap, the state after each token t of the call that snap_bits names (bit t) also goes to snap, the j-th such
/// state at snap + j * snap_stride floats (a window's snapshots: Engine::snap_at_).
void ink_conv(const float* x, float* state, const float* w, const float* dyn, float* out, int channels, int heads,
              int taps, int rows, void* stream, float* snap = nullptr, size_t snap_stride = 0, unsigned snap_bits = 0);

/// One token's ink_conv() (no dyn, one head) and its state, out written, then the stream acc updated: mode 0
/// acc = acc + out * a (axpy()), mode 1 keep = acc and then acc = out, mode 2 acc = out.  The same bits.
void ink_conv_add(const float* x, float* state, const float* w, float* out, float* acc, float* keep, float a, int mode,
                  int channels, int taps, void* stream);
/// One token's routed experts' sum into fo (moe_sum() onto zeros: fo = 0 + the n_used rows of x weighted by w, each
/// weight first times scale[ids[k]] where scale is set, expert_weights()'s product), then ink_conv_add() of fo in
/// mode 0, in one launch with the same bits.
void ink_moe_conv_add(const float* x, const float* w, const float* scale, const int* ids, int n_used, float* fo,
                      float* state, const float* cw, float* out, float* acc, float a, int channels, int taps,
                      void* stream);

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
    const float* gate = nullptr;     ///< rows * n_head * dim: out times sigmoid(gate) (ink_mul_sigmoid()); null: none
    void* q8 = nullptr;              ///< one token: out also quantized here, native_quantize_q8_1()'s bytes
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

// ---- the DSpark drafter (Inkling2::guess_block()) ----

/// A row's position: what the card holds at at[2 * t] in a recorded launch, else pos0 + t.
struct DsPos {
    const long long* at = nullptr;
    long long pos0 = 0;
};
/// The target's four features of `rows` rows (tap + k * tap_stride, n values a row) mixed four ways: mix j =
/// rms(sum_k alpha[j][k] f_k) * norm[j]; mixes 0..2 to out (j * rows + t rows on), mix 3 to ring row pos % ring_rows.
struct DsMix {
    const float* tap = nullptr;
    size_t tap_stride = 0;
    const float* alpha = nullptr;   ///< [4][4], each row a softmax over the features
    const float* norm[4] = {};
    float* out = nullptr;
    float* ring = nullptr;
    int ring_rows = 0;
    DsPos pos;
};
void ds_mix(const DsMix& m, int n, int rows, float eps, void* stream);
/// Each head of 128 values of `rows` rows of x RMS-normed with w and rotated at its row's position (NeoX halves,
/// base theta): back into x, or with cache into its fp16 cell pos % cap (heads * 128 halves a cell), v's heads to
/// vcache's beside it.
void ds_rope(float* x, int heads, int rows, const float* w, float eps, float theta, const DsPos& p, uint16_t* cache,
             int64_t cap, const float* v, uint16_t* vcache, void* stream);
/// Each of `rows` query rows' heads over the cache's positions [lo, hi), all of them (no causal mask); score has
/// rows * heads rows of score_stride floats of scratch.
void ds_attend(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t cap, int64_t lo, int64_t hi,
               int rows, int heads, int kv_heads, float scale, float* score, size_t score_stride, float* out,
               void* stream);
/// g = silu(g) * u
void ds_swiglu(float* g, const float* u, size_t n, void* stream);
/// rows 1 .. rows - 1 of x (n values a row) = mask
void ds_mask_rows(float* x, const float* mask, int n, int rows, void* stream);
/// in = [state ; BF16 row *prev of w1] (rank values each)
void ds_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, float* in, int rank, void* stream);
/// The Markov head's step: raw = ctx_h + ctx_f + bias + sp (3 * rank); gate = sigmoid(raw[0:r]), state = gate *
/// state + (1 - gate) * tanh(raw[r:2r]), latent = tanh(raw[2r:3r]).
void ds_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const float* sp, float* state,
                    float* latent, int rank, void* stream);
/// A draft's pick from its scores cap tanh((base + fy + fb) / cap) over the first n_keep tokens: the lowest id of the
/// largest and its softmax probability into out, the id into prev too.  scratch: ds_pick_floats() floats.
void ds_pick(const float* base, const float* fy, const float* fb, int n_keep, float cap, float* scratch, Likeliest* out,
             int32_t* prev, void* stream);
size_t ds_pick_floats();
/// Q8_0 block b of each of n_rows rows of `blocks` blocks (fold), from the 32 rows of ft (ft[k][v], k < 32).
void ds_fold_q8(const float* ft, int n_rows, int blocks, int b, void* fold, void* stream);

}  // namespace strata::poly
