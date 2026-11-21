// src/poly/inkling2.cpp - the layers of Inkling 2 with a looped core.  The reference for the math is llama.cpp's
// inkling2 graph (src/models/inkling2.cpp and inkling2-loop.cpp).
#include "inkling2.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;   // the attention's scores of the tokens it takes at a time
const char* const kCtrlName[] = {"q", "k", "v", "ffn"};

}  // namespace

void Inkling2::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    tok_norm_ = out_query_ = Dev{};
}

void Inkling2::load_geometry() {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    // The file has no block that guesses ahead, and the layers leave no state to go back to after a wrong guess:
    // a token at a time.
    opt_.window = 1;
    opt_.draft = false;
    need(opt_.batch == 0, "Inkling 2 runs one sequence (--batch is not supported for it yet)");
    need(opt_.layer_split.empty(), "Inkling 2 runs on one card (--layer-split is not supported for it yet)");
    need(opt_.kv_format == KvFmt::kFp16, "Inkling 2's attention caches are fp16");
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto first_u = [&](const char* key) {   // a number, or a list's first where every layer has the same
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        if (v->type != MetaType::ARRAY) return (int) v->u;
        need((int) v->items.size() >= n_layer_, key);
        for (int il = 1; il < n_layer_; ++il) need(v->items[(size_t) il].u == v->items[0].u, key);
        return (int) v->items[0].u;
    };
    n_head_ = first_u("attention.head_count");
    n_kv_ = first_u("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention heads");
    need(head_dim_ > 0 && head_dim_ <= 128 && shape_i("attention.value_length") == head_dim_, "bad head size");
    n_rel_ = shape_i("attention.relative_length");
    need(n_rel_ > 0 && n_rel_ <= 32, "bad attention.relative_length");
    need(n_head_ / n_kv_ <= 4 && (head_dim_ == 64 || head_dim_ == 128), "attention heads that ink_attend() does not take");
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    log_floor_ = (float) shape_i("attention.log_scale_floor");
    log_alpha_ = shape_f("attention.log_scale_alpha");
    need(qk_clip_ > 0 && rel_clip_ > 0 && log_floor_ >= 1 && log_alpha_ >= 0, "bad clipping or scaling");
    window_ = shape_i("attention.sliding_window");
    need(window_ > 0, "bad attention.sliding_window");
    taps_ = shape_i("shortconv.l_cache");
    need(taps_ >= 2 && taps_ <= 8, "shortconv.l_cache is not 2 to 8");
    blk_ = shape_i("attn_res.block_size");
    need(blk_ > 0 && (n_layer_ - 1) / blk_ + 1 <= kBlocks, "bad attn_res.block_size");
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    situ_beta_ = shape_f("activation.situ_beta");
    situ_lb_ = shape_f("activation.situ_linear_beta");
    need(situ_beta_ > 0 && situ_lb_ >= 0, "bad activation.situ_beta");
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->num() : 0.f;
    const MetaValue* unpadded = find("vocab_size_unpadded");
    n_unpadded_ = unpadded ? (int) unpadded->u : n_vocab_;
    need(n_unpadded_ > 0 && n_unpadded_ <= n_vocab_, "bad vocab_size_unpadded");
    core_start_ = shape_i("loop.start");
    core_end_ = shape_i("loop.end");
    n_pass_ = shape_i("loop.count");
    rank_ = shape_i("loop.gate_rank");
    const MetaValue* varying = find("loop.varying");
    need(!varying || varying->u == 0, "a loop of a varying number of passes is not supported");
    need(n_pass_ >= 2 && n_pass_ <= 8 && rank_ > 0 && core_start_ > 0 && core_start_ < core_end_ &&
             core_end_ <= n_layer_ && core_start_ % blk_ == 0 && core_end_ % blk_ == 0,
         "bad loop");
    need((core_end_ - core_start_) / blk_ - 1 <= 1, "a loop core of more than two blocks is not supported");
    // every expert's scale goes on its joined gate and up product, whatever the experts' type
    joined_scales_ = true;
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    const MetaValue* pattern = find("attention.sliding_window_pattern");
    need(pattern && pattern->type == MetaType::ARRAY && (int) pattern->items.size() >= n_layer_,
         "attention.sliding_window_pattern has no value for every layer");
    slots_.assign((size_t) n_layer_ + (size_t) (n_pass_ - 1) * (core_end_ - core_start_), Slot{});
    for (int il = 0; il < n_layer_; ++il)
        for (int p = 0; p < (il >= core_start_ && il < core_end_ ? n_pass_ : 1); ++p)
            slots_[(size_t) slot_of(il, p)].il = il;
    swa_.assign((size_t) n_layer_, false);
    for (int il = 0; il < n_layer_; ++il) swa_[(size_t) il] = pattern->items[(size_t) il].u != 0;
}

void Inkling2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    const int64_t e = n_embd_, hq = (int64_t) n_head_ * head_dim_, kv = (int64_t) n_kv_ * head_dim_;

    if (il == 0) {
        tok_norm_ = upload(gg_->find("token_embd_norm.weight"), "token_embd_norm.weight", e, 1, kF32);
        out_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", e, 1, kF32);
    }
    B.swa = swa_[(size_t) il];
    L.fixed = true;    // the launches read the position on the card when they are recorded (head_key(), tail_key())
    B.wq = mat("attn_q.weight", e, hq);
    B.wk = mat("attn_k.weight", e, kv);
    B.wv = mat("attn_v.weight", e, kv);
    B.wo = mat("attn_output.weight", hq, e);
    B.gate = mat("attn_gate.weight", e, hq);
    B.rel = mat("attn_rel.weight", e, (int64_t) n_rel_ * n_head_);
    B.sink = mat("attn_sink_proj.weight", e, n_head_);
    B.q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    const TensorInfo* bias = tensor(il, "attn_rel_bias.weight");
    if (!bias || bias->shape.size() != 2 || (int64_t) bias->shape[0] != n_rel_)
        throw std::runtime_error(blk(il, "attn_rel_bias.weight") + " is " + (bias ? shape_of(*bias) : "missing"));
    B.extent = (int) bias->shape[1];
    if (B.swa && B.extent != window_)
        throw std::runtime_error(blk(il, "attn_rel_bias.weight") + " does not have a column a position of the window");
    B.rel_bias = f32("attn_rel_bias.weight", n_rel_, B.extent);
    B.k_conv = f32("attn_k_conv.weight", taps_, kv);
    B.v_conv = f32("attn_v_conv.weight", taps_, kv);
    B.a_conv = f32("attn_post_conv.weight", taps_, e);
    B.f_conv = f32("ffn_post_conv.weight", taps_, e);
    if (B.swa) {
        B.attn_norm = f32("attn_norm.weight", e);
    } else {
        B.q_norm = f32("attn_q_inp_norm.weight", e);
        B.k_norm = f32("attn_k_inp_norm.weight", e);
        B.v_norm = f32("attn_v_inp_norm.weight", e);
        B.k_dyn = mat("attn_k_conv_proj.weight", e, (int64_t) taps_ * n_kv_);
        B.v_dyn = mat("attn_v_conv_proj.weight", e, (int64_t) taps_ * n_kv_);
    }
    if (il > 0) B.attn_query = f32("attn_res_query.weight", e, B.swa ? 1 : 3);
    B.ffn_query = f32("ffn_res_query.weight", e);
    B.ffn_norm = f32("ffn_norm.weight", e);
    if (il >= core_start_ && il < core_end_) {
        B.msg_query = f32("loop_msg_query.weight", e, kCtrl);
        for (int c = 0; c < kCtrl; ++c) {
            const std::string pre = std::string("loop_") + kCtrlName[c] + "_ctrl_";
            B.ctrl[c].down = f32((pre + "down.weight").c_str(), 2 * e, rank_);
            B.ctrl[c].gain = f32((pre + "gain.weight").c_str(), rank_, e);
            B.ctrl[c].forget = f32((pre + "forget.weight").c_str(), rank_, e);
        }
    }
    load_moe(il, false);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (b) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

size_t Inkling2::recurrent_floats() const { return slots_.size() * past_floats(); }

int Inkling2::widest() const { return std::max(n_head_ * head_dim_, 2 * n_embd_); }

// Every application of a layer has caches of its own: a full layer's keep a row a position, a window layer's a
// ring of the rows a call can read.  The convolutions' past inputs are in state_.
void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (Slot& s : slots_) {
        if (!runs(s.il)) continue;
        s.past = st;
        st += past_floats();
        if (blocks_[(size_t) s.il].swa) {
            s.keys = (uint16_t*) seq_alloc((size_t) ring_ * row, "the attention cache");
            s.vals = (uint16_t*) seq_alloc((size_t) ring_ * row, "the attention cache");
        } else {
            s.keys = (uint16_t*) kv_alloc(row, 1);
            s.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Inkling2::assign_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    for (float*& b : bl_) b = fbuf(e);
    bank_ = fbuf(e);
    bank_out_ = fbuf(e);
    for (float*& b : in_) b = fbuf(e);
    for (float*& b : xin_) b = fbuf(e);
    for (float*& b : msg_) b = fbuf(e);
    cat_ = fbuf(2 * e);
    ch_ = fbuf(n * rank_);
    gain_ = fbuf(e);
    forget_ = fbuf(e);
    mixf_ = fbuf(e);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    vraw_ = fbuf(kv);
    v_ = fbuf(kv);
    rr_ = fbuf(n * n_head_ * n_rel_);
    sink_ = fbuf(n * n_head_);
    gate_ = fbuf(heads);
    kdyn_ = fbuf(n * taps_ * n_kv_);
    vdyn_ = fbuf(n * taps_ * n_kv_);
    att_ = fbuf(heads);
    aoraw_ = fbuf(e);
    fc_ = fbuf(e);
    // a token's scores of the whole context, and what ink_attend() keeps beside them
    score_n_ = std::max(kScoreFloats, ink_attend_floats(n_head_, n_kv_, head_dim_, opt_.max_context));
    score_ = fbuf(score_n_);
    zeros_ = fbuf((size_t) n_head_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

void Inkling2::copy_rows(const float* from, float* to, int n) {
    ck(cudaMemcpyAsync(to, from, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "rows");
}

// The first `blocks` residual streams and the current one, from float `at` of each.
InkStates Inkling2::streams_of(int blocks, size_t at) const {
    InkStates s;
    for (int i = 0; i < blocks; ++i) s.p[s.n++] = bl_[i] + at;
    s.p[s.n++] = r_ + at;
    return s;
}

// A loop controller: out = cur - tanh(forget) * cur + tanh(gain) * msg, the two gates from cur and msg.  out may
// be cur.
void Inkling2::control(const float* cur, const float* msg, const Ctrl& c, float* out, int n) {
    ink_cat_norm(cur, msg, cat_, n_embd_, n, st_);
    lin(c.down, cat_, ch_, n);
    lin(c.gain, ch_, gain_, n);
    lin(c.forget, ch_, forget_, n);
    ink_gate(cur, msg, gain_, forget_, out, (size_t) n * n_embd_, st_);
}

// A layer from its input to its router's copy: the attention, the block's residual, and the experts' input.
void Inkling2::layer_head(Layer& L, int il, int n) {
    if (il == lo_) pass_ = 0;   // a call starts at its first layer
    Block& B = blocks_[(size_t) il];
    Slot& S = slots_[(size_t) slot_of(il, pass_of(il))];
    const size_t ne = (size_t) n * n_embd_;
    const int kv = n_kv_ * head_dim_, hq = n_head_ * head_dim_;

    if (il == 0) {   // the tokens' embeddings, normed
        rms_norm(r_, fp(tok_norm_), mixf_, n_embd_, n, rms_eps_, st_);
        copy_rows(mixf_, r_, n);
    }
    // A later pass of the core starts again from the stream the core's first block started from, with the pass
    // before it as its bank: that pass's later block (if the core has two) and its output.
    const int core_block = core_start_ / blk_;
    const int n_bank = (core_end_ - core_start_) / blk_ - 1;
    if (il == core_start_ && pass_ > 0) {
        if (n_bank > 0) copy_rows(bl_[core_block + 1], bank_, n);
        copy_rows(r_, bank_out_, n);
        copy_rows(bl_[core_block], r_, n);
    }

    // the attention's inputs: a mix of the streams so far, one for a window layer and one each for Q, K and V for
    // a full one
    const float* in[3] = {r_, r_, r_};
    if (il > 0) {
        float* out[3] = {in_[0], in_[1], in_[2]};
        ink_mix(streams_of((il - 1) / blk_ + 1, 0), fp(B.attn_query), B.swa ? 1 : 3, out, n_embd_, n, st_);
        in[0] = in_[0];
        in[1] = B.swa ? in_[0] : in_[1];
        in[2] = B.swa ? in_[0] : in_[2];
    }
    const bool looped = pass_of(il) > 0;   // a later pass of the core: its inputs take the pass before's messages
    if (looped) {
        InkStates bank;
        if (n_bank > 0) bank.p[bank.n++] = bank_;
        bank.p[bank.n++] = bank_out_;
        ink_mix(bank, fp(B.msg_query), kCtrl, msg_, n_embd_, n, st_);
        // Q's last: a window layer's three inputs are one buffer until the messages make them differ
        for (int j = 2; j >= 0; --j) {
            control(in[j], msg_[j], B.ctrl[j], in_[j], n);
            in[j] = in_[j];
        }
    }
    const float* x[3];
    if (B.swa) {
        rms_norm(in[0], fp(B.attn_norm), xin_[0], n_embd_, n, rms_eps_, st_);
        x[0] = x[1] = x[2] = xin_[0];
        if (looped)
            for (int j = 1; j < 3; ++j) {
                rms_norm(in[j], fp(B.attn_norm), xin_[j], n_embd_, n, rms_eps_, st_);
                x[j] = xin_[j];
            }
    } else {
        const Dev* norm[3] = {&B.q_norm, &B.k_norm, &B.v_norm};
        for (int j = 0; j < 3; ++j) {
            rms_norm(in[j], fp(*norm[j]), xin_[j], n_embd_, n, rms_eps_, st_);
            x[j] = xin_[j];
        }
    }
    dbg("attn_in", il, xin_[0], n_embd_);

    lin(B.wq, x[0], q_, n);
    lin(B.wk, x[1], kraw_, n);
    lin(B.wv, x[2], vraw_, n);
    lin(B.rel, x[0], rr_, n);
    lin(B.sink, x[0], sink_, n);
    lin(B.gate, x[0], gate_, n);
    const float *kd = nullptr, *vd = nullptr;
    if (!B.swa) {
        lin(B.k_dyn, x[1], kdyn_, n);
        lin(B.v_dyn, x[2], vdyn_, n);
        kd = kdyn_;
        vd = vdyn_;
    }
    ink_conv(kraw_, past_k(S), fp(B.k_conv), kd, k_, kv, n_kv_, taps_, n, st_);
    ink_conv(vraw_, past_v(S), fp(B.v_conv), vd, v_, kv, n_kv_, taps_, n, st_);
    const float alpha = B.swa ? 0.f : log_alpha_;
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens' positions from what at_ holds
    ink_clip(q_, fp(B.q_clip), qk_clip_, head_dim_, n_head_, n, pos_, alpha, log_floor_, st_, at);
    ink_clip(k_, nullptr, qk_clip_, head_dim_, n_kv_, n, pos_, 0.f, 1.f, st_);
    ink_head_rms(v_, head_dim_, n_kv_, n, rms_eps_, st_);
    ink_clip(rr_, fp(B.rel_clip), rel_clip_, n_rel_, n_head_, n, pos_, alpha, log_floor_, st_, at);
    if (at) {
        store_f16_at(k_, S.keys, kv, at, n, B.swa ? ring_ : 0, st_);
        store_f16_at(v_, S.vals, kv, at, n, B.swa ? ring_ : 0, st_);
    } else if (B.swa) {
        ink_store_ring(k_, S.keys, kv, pos_, n, ring_, st_);
        ink_store_ring(v_, S.vals, kv, pos_, n, ring_, st_);
    } else {
        store_f16(k_, S.keys, kv, pos_, n, st_);
        store_f16(v_, S.vals, kv, pos_, n, st_);
    }
    InkAttn a;
    a.q = q_;
    a.r = rr_;
    a.sink = sink_;
    a.keys = S.keys;
    a.vals = S.vals;
    a.rel_bias = fp(B.rel_bias);
    a.out = att_;
    a.score = score_;
    a.score_n = score_n_;
    a.pos = pos_;
    a.at = at;
    a.room = at ? room(n) : 0;
    a.rows = n;
    a.n_head = n_head_;
    a.n_kv = n_kv_;
    a.dim = head_dim_;
    a.n_rel = n_rel_;
    a.extent = B.extent;
    a.window = B.swa ? window_ : 0;
    a.ring = B.swa ? ring_ : 0;
    a.scale = 1.f / std::sqrt((float) head_dim_);
    if (n >= kStageRows) {
        // a prompt batch: the engine's products on the tensor cores, which round the queries and the softmax's
        // weights to halves; the sink's own term of the softmax has score 0 (zeros_)
        AttnBias bias;
        bias.r = rr_;
        bias.sink = sink_;
        bias.table = fp(B.rel_bias);
        bias.n_rel = n_rel_;
        bias.extent = B.extent;
        ck(cudaMemsetAsync(zeros_, 0, (size_t) n_head_ * sizeof(float), cs(st_)), "attention");
        attend_batch(blas_, q_, S.keys, S.vals, pos_, n, score_, score_n_, awork_, att_, head_dim_, n_head_, n_kv_,
                     a.scale, st_, a.window, a.ring, false, {}, {}, nullptr, nullptr, 0, zeros_, nullptr, &bias);
    } else {
        ink_attend(a, st_);
    }
    ink_mul_sigmoid(att_, gate_, (size_t) n * hq, st_);
    lin(B.wo, att_, aoraw_, n);
    ink_conv(aoraw_, past_a(S), fp(B.a_conv), nullptr, ao_, n_embd_, 1, taps_, n, st_);
    dbg("attn", il, ao_, n_embd_);

    // the first layer of a block starts a new stream from the attention's output; the one it had joins the mixes
    if (il % blk_ == 0) {
        if (!(il == core_start_ && looped)) copy_rows(r_, bl_[il / blk_], n);
        copy_rows(ao_, r_, n);
    } else {
        axpy(ao_, 1.f, r_, (int) ne, true, st_);
    }

    // the experts' input
    float* mixed[1] = {mixf_};
    ink_mix(streams_of(il / blk_ + 1, 0), fp(B.ffn_query), 1, mixed, n_embd_, n, st_);
    if (looped) control(mixf_, msg_[3], B.ctrl[3], mixf_, n);
    kernels::native_add_rms_norm_q8_1(mixf_, nullptr, fp(B.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_,
                                      sum_threads(n_embd_), st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The experts' sum through its convolution, onto the stream.
void Inkling2::layer_tail(Layer&, int il, int n) {
    const Slot& S = slots_[(size_t) slot_of(il, pass_of(il))];
    ink_conv(fo_, past_f(S), fp(blocks_[(size_t) il].f_conv), nullptr, fc_, n_embd_, 1, taps_, n, st_);
    axpy(fc_, 1.f, r_, n * n_embd_, true, st_);
}

// After the core's last layer, its first again, until the passes are done.  Called for every layer a call runs,
// replayed or not, so the pass is right for the recordings' keys.
int Inkling2::next_layer(int il) {
    if (il == core_end_ - 1) {
        if (pass_ + 1 < n_pass_) {
            ++pass_;
            return core_start_;
        }
        pass_ = 0;
    }
    return il + 1;
}

// A layer's recordings: one a pass of the core, and for a full layer one a size of the rows it reads.
int Inkling2::head_key(int il, int n) const {
    return pass_of(il) * 32 + (blocks_[(size_t) il].swa ? 0 : room_key(n));
}

int Inkling2::tail_key(int il, int) const { return pass_of(il); }

void Inkling2::output_rows(int first, int rows, float* h) {
    const size_t at = (size_t) first * n_embd_;
    float* mixed[1] = {mixf_ + at};
    ink_mix(streams_of((n_layer_ - 1) / blk_ + 1, at), fp(out_query_), 1, mixed, n_embd_, rows, st_);
    rms_norm(mixf_ + at, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// The scores' soft cap, and the output matrix's rows past the vocabulary out of every pick.
void Inkling2::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f || n_unpadded_ < n_vocab_)
        ink_softcap(logits, n_unpadded_, n_vocab_, rows, softcap_ > 0.f ? softcap_ : INFINITY, st_);
}

// What llama.cpp's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights
// from the bare gates, divided by their sum and scaled.
void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = sum + 1e-20f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr, nullptr, true,
               w_scale_, ids, w, st_);
    return true;
}

// No shared expert: the routed experts' sum starts from nothing.
void Inkling2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// ---- a sequence's state and rows ----

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).  A position before the sequence's start has no run.
template <class F>
void Inkling2::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What the window slots have of the sequence: the rows the next token attends, keys then values.
size_t Inkling2::extra_bytes() const {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    size_t n = 0;
    for (const Slot& s : slots_) n += blocks_[(size_t) s.il].swa ? 1 : 0;
    return n * 2 * (size_t) keep_ * row;
}

// A layer's slots: each one's convolution inputs at their place in the state, and a window slot's rows behind the
// state, after the window slots before it.
void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), whole = 2 * (size_t) keep_ * row;
    size_t rings = 0;
    for (const Slot& s : slots_) {
        const bool swa = blocks_[(size_t) s.il].swa;
        if (s.il == il && s.past) {
            ck(cudaMemcpyAsync(dst + (size_t) (s.past - state_) * sizeof(float), s.past,
                               past_floats() * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
               "save state");
            if (swa) {
                uint8_t* out = dst + state_n_ * sizeof(float) + rings * whole;
                for (const uint16_t* cache : {s.keys, s.vals}) {
                    std::memset(out, 0, (size_t) keep_ * row);   // the places of positions before the start
                    window_runs(pos, [&](size_t place, size_t at, size_t m) {
                        ck(cudaMemcpyAsync(out + place * row, (const uint8_t*) cache + at * row, m * row,
                                           cudaMemcpyDeviceToHost, cs(st_)),
                           "save state");
                    });
                    out += (size_t) keep_ * row;
                }
            }
        }
        rings += swa ? 1 : 0;
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (const Slot& s : slots_) {
        if (!blocks_[(size_t) s.il].swa) continue;
        for (uint16_t* cache : {s.keys, s.vals}) {
            if (cache)
                window_runs(pos, [&](size_t place, size_t at, size_t m) {
                    ck(cudaMemcpy((uint8_t*) cache + at * row, src + place * row, m * row, cudaMemcpyHostToDevice),
                       "load state");
                });
            src += (size_t) keep_ * row;
        }
    }
}

// The full slots' rows of the whole context, keys then values a slot.
size_t Inkling2::card_row_bytes() const {
    size_t n = 0;
    for (const Slot& s : slots_) n += blocks_[(size_t) s.il].swa ? 0 : 1;
    return n * 2 * (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (const Slot& s : slots_) {
        if (blocks_[(size_t) s.il].swa || !s.keys || n == 0) continue;
        for (const uint16_t* cache : {s.keys, s.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, (size_t) n * row,
                          cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * row;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (const Slot& s : slots_) {
        if (blocks_[(size_t) s.il].swa || !s.keys || n == 0) continue;
        for (uint16_t* cache : {s.keys, s.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row,
                          (size_t) n * row, cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}

}  // namespace strata::poly
