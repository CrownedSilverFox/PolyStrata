// src/poly/mmq_own.cu - mmq.hpp's prompt products without llama.cpp, on the kernel library's two grouped kernels
// (src/kernels/cuda).  The weights stay in their GGUF blocks and are decoded into shared memory a tile at a time.
// mmq.cu is the llama.cpp form of the same class (-DSTRATA_POLY_MMQ=ON); one of the two is built.
//
// On Ampere and newer, a type the int8 decoders have and rows of whole stages of 256 values go to the int8 tensor
// cores (gemm_iq_i8_grouped, i8_mma.cuh): the activations are int8 with a scale per 32 values, MMQ's numbers, and a
// matrix may have any number of rows.  quantize() and mul() choose by the type and the row length alone, so the rows
// one wrote are the rows the other reads.
//
// Everything else goes to the FP16 kernel (gemm_iq_f16_grouped): the activations FP16 rows, the sums FP32.  What
// differs there from MMQ for the callers:
//   - the activations are halves, not Q8_1 blocks.  A half holds 11 bits from 6e-5 to 65504, and a model's rows are
//     not all in that range (Flint's are far above it, a hidden row behind NVFP4's scales far below), so quantize()
//     multiplies each row by the power of two that puts its largest value near 1024, which is exact, and mul()
//     divides the row's products by it again.  The engine's guard for MMQ's half sums has nothing to do here;
//   - a row's product does not depend on the launch it is in (split_exact()): each output value is its own sum
//     over the row's tiles in their order;
//   - no kernel for the cards before Volta, for a matrix whose rows are not whole tiles of 128, for a type the
//     dequantizer does not have (Q2_K, MXFP4, IQ1_S, the float types) or where a tile's shared memory is more than
//     the card gives a block (Turing: 64 KiB): fits() is false there and the engine reads such a batch a row at a time.
#include "mmq.hpp"

#if defined(GGML_USE_HIP) || defined(STRATA_USE_HIP) || defined(__HIPCC__)

// HIP: the grouped kernel is CUDA's.  No prompt product here, so every batch is read a row at a time.
#include <stdexcept>

namespace strata::poly {
Mmq::Mmq() {}
Mmq::~Mmq() {}
bool Mmq::fits(int, int64_t, int64_t) { return false; }
size_t Mmq::q8_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) cols * 2; }
bool Mmq::half_sums(int) { return false; }
bool Mmq::split_exact(int, int64_t) { return true; }
bool Mmq::reads_by_index(int, int64_t) { return false; }
void Mmq::quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {}
void Mmq::mul(const void*, int, int64_t, int64_t, size_t, int, const void*, const int32_t*, const int32_t*, int64_t, int64_t,
              float*, int64_t, void*, const int32_t*) {
    throw std::runtime_error("strata-poly prompt product: not in a HIP build");
}
}  // namespace strata::poly

#else

#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int kTileRows = 128;        // gemm_iq_f16_grouped's WG_BN: weight rows a block
constexpr float kHalfMax = 65504.f;   // the largest finite half

// The types iq_kernels.cu's dq_dispatch dequantizes (its is_iq(), a build with STRATA_Q6K_EXPERTS, the default).
bool dequantized(int t) {
    switch (t) {
        case 2: case 3: case 6: case 7: case 8:                       // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
        case 11: case 12: case 13: case 14:                           // Q3_K Q4_K Q5_K Q6_K
        case 16: case 17: case 18: case 20: case 21: case 22: case 23: case 29:   // the i-quants
        case 40: case 42:                                             // NVFP4, Q2_0
            return true;
        default:
            return false;
    }
}

// values of a row the kernel reads at a time: whole superblocks of 256, or blocks of 128 for the two types that have them
bool whole_tiles(int t, int64_t cols) {
    if (t == 42 || (t == 20 && cols % 256 != 0)) return cols % 128 == 0;
    return cols % 256 == 0;
}

// shared memory of a block (the kernel's own sum: BN weight rows of WK + 8 halves, BM = 64 rows of XK + 8 = 136)
size_t tile_bytes(int t, int64_t cols) {
    const int wk = (t == 42 || (t == 20 && cols % 256 != 0)) ? 128 : 256;
    return (size_t) kTileRows * (wk + 8) * 2 + (size_t) 64 * (128 + 8) * 2;
}

struct Card {
    int major = 0;
    size_t shared = 0;
};

// the current card's (a layer split asks for each of its cards): read once a card
Card card() {
    static Card known[64];
    static bool read[64];
    int dev = 0, v = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return Card{};
    if (!read[dev]) {
        Card c;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess) c.major = v;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess) c.shared = (size_t) v;
        known[dev] = c;
        read[dev] = true;
    }
    return known[dev];
}

// a block a row; the rows go over the grid's two sides, which hold 65,535 each
dim3 rows_grid(int64_t rows) {
    const unsigned gx = (unsigned) (rows < 32768 ? rows : 32768);
    return dim3(gx, (unsigned) ((rows + gx - 1) / gx));
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("strata-poly prompt product: ") + what + ": " + cudaGetErrorString(e));
}

// Row i of xq = row ids[i] of x (row i without ids) as halves, times a power of two f that puts the row's largest
// value in [1024, 2048): the multiplication is exact, so what is lost is a half's own rounding and nothing to the
// row's size.  back[i] = 1 / f, which mul() multiplies the row's products by.  A row of zeros, or one with a value
// that is not finite, is left as it is (and a value beyond a half's range saturated).
__global__ void to_halves(const float* __restrict__ x, const int32_t* __restrict__ ids, __half* __restrict__ xq, int cols,
                          int64_t ld, int64_t rows, float* __restrict__ back) {
    const int64_t row = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows) return;
    const float* src = x + (ids ? (int64_t) ids[row] : row) * ld;
    __shared__ float most[256];
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(src[c]));   // (a NaN is passed over by fmaxf)
    most[threadIdx.x] = m;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) most[threadIdx.x] = fmaxf(most[threadIdx.x], most[threadIdx.x + step]);
        __syncthreads();
    }
    int e = 11;
    if (most[0] > 0.f && most[0] <= 3.0e38f) {
        frexpf(most[0], &e);                      // most = m * 2^e, m in [0.5, 1)
        e = max(-100, min(127, e));
    }
    const float f = ldexpf(1.f, 11 - e);
    __half* dst = xq + row * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        dst[c] = __float2half(fminf(fmaxf(src[c] * f, -kHalfMax), kHalfMax));
    if (threadIdx.x == 0) back[row] = ldexpf(1.f, e - 11);
}

// Row ids[i] of dst (row i without ids) = row i of y times back[i], for the rows of this launch's matrices only:
// [bounds[0], bounds[n]).  The other rows of y are an earlier launch's (another run of experts, or a mixed layer's
// other half), and dst's are not this launch's to write.  y may be dst itself where no row moves.
__global__ void finish_rows(const float* __restrict__ y, int64_t ldy, const int32_t* __restrict__ ids,
                            const float* __restrict__ back, float* dst, int64_t ld_dst, int cols, int64_t rows,
                            const int32_t* __restrict__ bounds, int n) {
    const int64_t row = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (row >= rows || row < bounds[0] || row >= bounds[n]) return;
    const float* src = y + row * ldy;
    float* out = dst + (ids ? (int64_t) ids[row] : row) * ld_dst;
    const float b = back[row];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = src[c] * b;
}

// the int8 kernel takes a matrix of this type with rows this long, whatever their number
bool int8_rows(int t, int64_t cols) { return cols <= 0x7fffffff && kernels::gemm_iq_i8_takes(t, (int) cols); }

// the rows' halves, then a float a row (back)
size_t halves_bytes(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols * 2 + 3) / 4 * 4; }

// One prompt path's buffers on the card: where a single matrix's two bounds lie, and the products of a launch
// whose rows go elsewhere.
struct Ctx {
    int32_t* bounds = nullptr;
    int32_t bounds_host[2] = {0, 0};
    float* y = nullptr;
    size_t y_floats = 0;
};

}  // namespace

Mmq::Mmq() {
    Ctx* c = new Ctx;
    if (cudaMalloc(&c->bounds, 2 * sizeof(int32_t)) != cudaSuccess) {
        delete c;
        throw std::runtime_error("strata-poly prompt product: no memory on the card for two numbers");
    }
    ctx_ = c;
}

Mmq::~Mmq() {
    Ctx* c = (Ctx*) ctx_;
    cudaFree(c->bounds);
    cudaFree(c->y);
    delete c;
}

bool Mmq::fits(int ggml_type, int64_t w_rows, int64_t w_cols) {
    if (int8_rows(ggml_type, w_cols)) return true;
    const Card c = card();
    return c.major >= 7 && dequantized(ggml_type) && w_rows % kTileRows == 0 && whole_tiles(ggml_type, w_cols) &&
           tile_bytes(ggml_type, w_cols) <= c.shared;
}

size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    return (halves_bytes(rows, cols) + (size_t) rows * sizeof(float) + 255) / 256 * 256;
}

bool Mmq::half_sums(int) { return false; }

bool Mmq::split_exact(int, int64_t) { return true; }

bool Mmq::reads_by_index(int ggml_type, int64_t w_cols) { return int8_rows(ggml_type, w_cols); }

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows <= 0) return;
    if (int8_rows(ggml_type, cols)) return kernels::gemm_iq_i8_quantize(x, ids, ld, rows, (int) cols, xq, stream);
    float* back = (float*) ((char*) xq + halves_bytes(rows, cols));
    to_halves<<<rows_grid(rows), 256, 0, (cudaStream_t) stream>>>(x, ids, (__half*) xq, (int) cols, ld, rows, back);
    ck(cudaGetLastError(), "the rows as halves");
}

void Mmq::mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
              int64_t ld_dst, void* stream, const int32_t* from) {
    if (n <= 0 || total_rows <= 0) return;
    Ctx& c = *(Ctx*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    if (!bounds) {   // one matrix takes every row
        if (n != 1) throw std::runtime_error("strata-poly prompt product: several matrices need their bounds");
        c.bounds_host[0] = 0;
        c.bounds_host[1] = (int32_t) total_rows;
        ck(cudaMemcpyAsync(c.bounds, c.bounds_host, sizeof c.bounds_host, cudaMemcpyHostToDevice, s), "a matrix's bounds");
        bounds = c.bounds;
    }
    if (int8_rows(ggml_type, w_cols)) {   // straight to dst's rows
        if (!kernels::gemm_iq_i8_grouped(ggml_type, w, stride, (int) w_rows, (int) w_cols, xq, total_rows, bounds, n,
                                         (int) max_rows, ids, dst, ld_dst, stream, from))
            throw std::runtime_error("strata-poly prompt product: the int8 kernel did not start for tensor type " +
                                     std::to_string(ggml_type) + " with [" + std::to_string(w_rows) + ", " +
                                     std::to_string(w_cols) + "]");
        return;
    }
    if (from) throw std::logic_error("strata-poly prompt product: rows by index need the int8 kernel");
    // straight into dst where row i is row i (finish_rows then works in place); else by way of a buffer of this path's own
    float* y = dst;
    int64_t ldy = ld_dst;
    if (ids || ld_dst % 4 != 0) {
        ldy = (w_rows + 3) / 4 * 4;
        const size_t need = (size_t) total_rows * (size_t) ldy;
        if (need > c.y_floats) {
            ck(cudaStreamSynchronize(s), "waiting for the launches before");   // they may still read the old buffer
            cudaFree(c.y);
            c.y = nullptr;
            c.y_floats = 0;
            ck(cudaMalloc(&c.y, need * sizeof(float)), "memory for a launch's products");
            c.y_floats = need;
        }
        y = c.y;
    }
    if (!kernels::gemm_iq_f16_grouped(ggml_type, w, stride, (int) w_rows, (int) w_cols, xq, (int) w_cols, bounds, n,
                                      (int) max_rows, y, (int) ldy, stream))
        throw std::runtime_error("strata-poly prompt product: no kernel for tensor type " + std::to_string(ggml_type) +
                                 " with [" + std::to_string(w_rows) + ", " + std::to_string(w_cols) + "]");
    const float* back = (const float*) ((const char*) xq + halves_bytes(total_rows, w_cols));
    finish_rows<<<rows_grid(total_rows), 256, 0, s>>>(y, ldy, ids, back, dst, ld_dst, (int) w_rows, total_rows, bounds, n);
    ck(cudaGetLastError(), "the products to their rows");
}

}  // namespace strata::poly

#endif
