// src/poly/inkling2.hpp - Thinking Machines' Inkling 2 with a looped core (llama.cpp architecture "inkling2_loop"):
// Inkling 2 Mini.
//
// 24 layers of weights make 48 layer applications: the layers before the core run once, the core's run
// loop.count times, each pass from the output of the layers before it, and the layers after it once.  A layer of a
// later pass mixes messages from the pass before into its inputs through small gated controllers.  Every application
// has attention rows and convolution state of its own (a "slot"); the routed experts are the layer's, whatever the
// pass.
//
// A layer: attention with no rotary embedding (a learned bias by distance and a sink term in the softmax), short
// causal convolutions over K, V, the attention's output and the FFN's, and routed experts with the SiTU activation.
// Every block_size layers start a new residual stream; a layer's inputs are a learned mix of the streams so far.
// Most layers attend a window of the last tokens, a few all of them.
#pragma once

#include "engine.hpp"
#include "inkling2_kernels.hpp"

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    size_t extra_bytes() const override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    int next_layer(int il) override;
    int head_key(int il, int n) const override;
    int tail_key(int il, int n) const override;
    bool reads_by_layer() const override { return false; }
    void output_rows(int first, int rows, float* h) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;

private:
    static constexpr int kCtrl = 4;     ///< a looped layer's controllers: Q, K, V and the FFN
    static constexpr int kBlocks = 4;   ///< the most residual streams before the current one

    /// A loop controller: [normed input | normed message] down to the gate's rank, and its gain and forget gates.
    struct Ctrl {
        Dev down, gain, forget;
    };
    /// A layer's weights beside the ones every architecture has (Layer).
    struct Block {
        bool swa = false;   ///< attends the window; else every token so far
        int extent = 0;     ///< the distances its bias table has
        Dev wq, wk, wv, wo, gate, rel, sink;
        Dev q_clip, rel_bias, rel_clip;
        Dev k_conv, v_conv, a_conv, f_conv;
        Dev attn_norm;                       ///< a window layer's
        Dev q_norm, k_norm, v_norm;          ///< a full layer's
        Dev k_dyn, v_dyn;                    ///< a full layer's convolution taps by the input
        Dev attn_query, ffn_query;           ///< the mixes over the residual streams
        Dev ffn_norm;
        // a core layer's
        Dev msg_query;
        Ctrl ctrl[kCtrl];
        // routed experts: the bias their pick adds to the gates
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };
    /// What one application of a layer keeps of the sequence.
    struct Slot {
        int il = 0;
        uint16_t* keys = nullptr;   ///< fp16, n_kv * head_dim a row: a row a position, or a window layer's ring
        uint16_t* vals = nullptr;
        float* past = nullptr;      ///< the convolutions' last inputs: K, V, the attention's output, the FFN's
    };

    int slot_of(int il, int pass) const {
        return pass == 0 ? il : n_layer_ + (pass - 1) * (core_end_ - core_start_) + (il - core_start_);
    }
    /// The pass layer il's application under way is of: the core's pass_, and 0 outside the core.
    int pass_of(int il) const { return il >= core_start_ && il < core_end_ ? pass_ : 0; }
    size_t past_floats() const { return (size_t) (taps_ - 1) * (2 * (size_t) n_kv_ * head_dim_ + 2 * (size_t) n_embd_); }
    float* past_k(const Slot& s) const { return s.past; }
    float* past_v(const Slot& s) const { return s.past + (size_t) (taps_ - 1) * n_kv_ * head_dim_; }
    float* past_a(const Slot& s) const { return s.past + (size_t) (taps_ - 1) * 2 * n_kv_ * head_dim_; }
    float* past_f(const Slot& s) const { return past_a(s) + (size_t) (taps_ - 1) * n_embd_; }
    InkStates streams_of(int blocks, size_t at) const;
    void control(const float* cur, const float* msg, const Ctrl& c, float* out, int n);
    bool control3(const float* const* cur, float* const* msg, const Ctrl* c, float* const* out, int n);
    bool project(const Block& B, const float* const* x, int n);
    void copy_rows(const float* from, float* to, int n);
    template <class F> void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer
    std::vector<Slot> slots_;     ///< one an application of a layer
    std::vector<bool> swa_;       ///< a layer attends the window
    Dev tok_norm_, out_query_;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rel_ = 0, taps_ = 0, blk_ = 0;
    int window_ = 0, ring_ = 0, keep_ = 0;
    int core_start_ = 0, core_end_ = 0, n_pass_ = 1, rank_ = 0;
    int n_unpadded_ = 0;
    float qk_clip_ = 0, rel_clip_ = 0, log_alpha_ = 0, log_floor_ = 1, softcap_ = 0;
    float w_scale_ = 0;
    int pass_ = 0;   ///< the core's pass the call under way is in

    // device scratch, a row a token
    float* bl_[kBlocks] = {};   ///< the residual streams the blocks started from
    float *bank_ = nullptr, *bank_out_ = nullptr;
    float* in_[3] = {};
    float* xin_[3] = {};
    float* msg_[kCtrl] = {};
    float *cat_ = nullptr, *ch_ = nullptr, *gain_ = nullptr, *forget_ = nullptr, *mixf_ = nullptr;
    float *cat3_[3] = {}, *ch3_[3] = {}, *gain3_[3] = {}, *forget3_[3] = {};   ///< control3()'s, kStageRows rows
    void* xq3_[3] = {};   ///< project()'s quantized inputs, a token's
    float *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr, *rr_ = nullptr,
          *sink_ = nullptr, *gate_ = nullptr, *kdyn_ = nullptr, *vdyn_ = nullptr, *att_ = nullptr, *aoraw_ = nullptr,
          *fc_ = nullptr, *score_ = nullptr, *zeros_ = nullptr, *awork_ = nullptr;
    size_t score_n_ = 0;
};

}  // namespace strata::poly
