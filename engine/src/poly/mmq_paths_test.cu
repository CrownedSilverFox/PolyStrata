// src/poly/mmq_paths_test.cu - Mmq::quantize + Mmq::mul against a double-precision product of ggml's own dequantized
// weights, for the types strata-poly's models keep matrices in: one dense matrix, and four experts of a launch with
// uneven rows, an empty one, and their products scattered to other rows (bounds and ids, the bounds mirrored on the
// host as the engine does).  The path is the card's (int8 tiles, fp16 tiles, or the BLAS path; STRATA_POLY_PROMPT_BLAS=1
// forces the BLAS path), so a run prints that path's error: run it both ways to compare them.  The activations are
// normal values with a few channels 30 times larger, and rows of different sizes, as a transformer's are.
//
// usage: poly_mmq_paths_test   (exit 0: every product within 2% RMS of the reference; 77: no card)
#include "mmq.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using strata::poly::Mmq;

namespace {

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

struct Err { double rms = 0, top = 0; };

// y against ref over the rows listed (each `cols` long, ld apart)
Err compare(const std::vector<float>& y, const std::vector<double>& ref, int64_t rows, int64_t cols, int64_t ld) {
    double se = 0, sr = 0, me = 0, mr = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
            const double e = (double) y[(size_t) (r * ld + c)] - ref[(size_t) (r * cols + c)], v = ref[(size_t) (r * cols + c)];
            se += e * e;
            sr += v * v;
            me = std::max(me, std::fabs(e));
            mr = std::max(mr, std::fabs(v));
        }
    return {std::sqrt(se / std::max(sr, 1e-300)), me / std::max(mr, 1e-300)};
}

}  // namespace

int main() {
    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) {
        std::puts("poly_mmq_paths_test: no card, skipped");
        return 77;
    }
    const char* forced = std::getenv("STRATA_POLY_PROMPT_BLAS");
    std::printf("path: %s\n", forced && forced[0] == '1' ? "BLAS (STRATA_POLY_PROMPT_BLAS=1)" : "the card's");
    const int types[] = {8, 12, 14, 16, 18, 22, 23};
    constexpr int64_t kOut = 512, kIn = 2048, kTokens = 256, kExperts = 4;
    std::mt19937 rng(99);
    std::normal_distribution<float> normal(0.f, 1.f);
    // activations: a scale a row, a few outlier channels
    std::vector<float> x((size_t) (kTokens * kIn));
    std::vector<float> ch((size_t) kIn, 1.f);
    for (int64_t c = 0; c < kIn; c += 97) ch[(size_t) c] = 30.f;
    for (int64_t r = 0; r < kTokens; ++r) {
        const float s = std::exp(normal(rng));
        for (int64_t c = 0; c < kIn; ++c) x[(size_t) (r * kIn + c)] = normal(rng) * s * ch[(size_t) c];
    }
    // the experts' rows: 100, 0, 7 and the rest; row i's product goes to row perm[i]
    const int32_t bounds_h[kExperts + 1] = {0, 100, 100, 107, (int32_t) kTokens};
    std::vector<int32_t> perm((size_t) kTokens);
    for (int64_t i = 0; i < kTokens; ++i) perm[(size_t) i] = (int32_t) ((i * 37 + 11) % kTokens);
    float* dx = upload(x);
    int32_t* dperm = upload(perm);
    int32_t* dbounds = upload(std::vector<int32_t>(bounds_h, bounds_h + kExperts + 1));
    if (!dx || !dperm || !dbounds) return 1;
    Mmq mmq;
    mmq.mirror(dbounds, bounds_h, sizeof bounds_h);
    void* xq = nullptr;
    float* dy = nullptr;
    if (!ok(cudaMalloc(&xq, Mmq::q8_bytes(kTokens, kIn)), "cudaMalloc") ||
        !ok(cudaMalloc(&dy, (size_t) (kTokens * kOut) * sizeof(float)), "cudaMalloc"))
        return 1;
    int bad = 0;
    for (int t : types) {
        const auto ty = (ggml_type) t;
        if (!Mmq::fits(t, kOut, kIn)) {
            std::printf("%-8s no product on this card\n", ggml_type_name(ty));
            continue;
        }
        ggml_quantize_init(ty);
        const size_t mat = ggml_row_size(ty, kIn) * kOut;
        std::vector<float> wf((size_t) (kExperts * kOut * kIn)), imp((size_t) kIn);
        for (auto& v : wf) v = normal(rng) * 0.02f;
        for (auto& v : imp) v = 0.5f + std::fabs(normal(rng));
        std::vector<uint8_t> wq(mat * kExperts);
        ggml_quantize_chunk(ty, wf.data(), wq.data(), 0, kExperts * kOut, kIn, imp.data());
        std::vector<float> wd(wf.size());
        ggml_get_type_traits(ty)->to_float(wq.data(), wd.data(), (int64_t) wd.size());
        uint8_t* dw = upload(wq);
        if (!dw) return 1;
        // row i of a launch times its expert's matrix, in double
        auto reference = [&](int64_t row, int e, double* out) {
            const float* a = &x[(size_t) (row * kIn)];
            for (int64_t o = 0; o < kOut; ++o) {
                const float* w = &wd[(size_t) ((e * kOut + o) * kIn)];
                double s = 0;
                for (int64_t k = 0; k < kIn; ++k) s += (double) a[k] * w[k];
                out[o] = s;
            }
        };
        std::vector<double> ref((size_t) (kTokens * kOut));
        std::vector<float> y((size_t) (kTokens * kOut));
        // one dense matrix: every row, in place
        Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
        mmq.mul(dw, t, kOut, kIn, 0, 1, xq, nullptr, nullptr, kTokens, kTokens, dy, kOut, nullptr);
        if (!ok(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "dense product")) return 1;
        for (int64_t r = 0; r < kTokens; ++r) reference(r, 0, &ref[(size_t) (r * kOut)]);
        const Err dense = compare(y, ref, kTokens, kOut, kOut);
        // four experts: rows [bounds[e], bounds[e + 1]) of xq (row i of xq is row i of x), product of row i to row perm[i]
        cudaMemset(dy, 0, y.size() * sizeof(float));
        Mmq::quantize(dx, nullptr, xq, t, kIn, kIn, kTokens, nullptr);
        mmq.mul(dw, t, kOut, kIn, mat, kExperts, xq, dbounds, dperm, kTokens, 149, dy, kOut, nullptr);
        if (!ok(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost), "experts' product")) return 1;
        std::vector<float> back((size_t) (kTokens * kOut));
        for (int e = 0; e < kExperts; ++e)
            for (int64_t r = bounds_h[e]; r < bounds_h[e + 1]; ++r) {
                reference(r, e, &ref[(size_t) (r * kOut)]);
                std::copy_n(&y[(size_t) perm[(size_t) r] * kOut], kOut, &back[(size_t) (r * kOut)]);
            }
        const Err experts = compare(back, ref, kTokens, kOut, kOut);
        const bool fine = dense.rms < 0.02 && experts.rms < 0.02;
        std::printf("%-8s dense: RMS error %.2e, largest %.2e   experts: RMS error %.2e, largest %.2e  %s\n", ggml_type_name(ty),
                    dense.rms, dense.top, experts.rms, experts.top, fine ? "ok" : "WRONG");
        bad += !fine;
        cudaFree(dw);
    }
    ggml_quantize_free();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
