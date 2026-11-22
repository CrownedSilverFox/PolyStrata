// src/poly/logits_dump.cpp - the engine's logits for a list of token ids, as raw floats, so that two builds (or two
// settings: graphs on and off, a kernel and its replacement) can be compared byte for byte, or by KL against a
// reference card.  Run by hand; see kUsage for the options.
//
// The prompt goes in as strata-poly's --tokens does (main.cpp, once()): all but its last token in reads of
// max_read() tokens, then the last one alone, with logits.  Then each --decode token goes in alone (or, with
// --greedy N, N tokens each the likeliest of the logits before it, the lowest id among equals).  DIR/prompt_logits.f32
// gets the prompt's last logits, DIR/decode_logits.f32 a row a decode token (n_vocab() floats a row, or --vocab N's
// first N).  The ids fed and the five likeliest of each row go to stdout.  For identity runs, pin the expert cache:
// STRATA_GPU_EXPERTS=N (which also freezes it) and a cache-set hash on stderr to compare.
#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* const kUsage =
    "usage: poly_logits_dump MODEL.gguf --tokens ID,ID,...|@FILE [options]\n"
    "  --tokens ID,ID,...     the prompt's token ids; @FILE reads them from a file, apart by commas, white space or\n"
    "                         brackets (a JSON list)\n"
    "  --tokens-file FILE     the same as --tokens @FILE\n"
    "  --decode ID,ID,...|@FILE  tokens fed one at a time after the prompt, each with its logits\n"
    "  --greedy N             N tokens fed after the prompt, each the likeliest of the row before (after --decode's)\n"
    "  --out DIR              where the .f32 files go (default .)\n"
    "  --vocab N              write a row's first N scores (Inkling 2: 200058, the unpadded vocabulary)\n"
    "  --max-context N        the engine's context (default 4096)\n"
    "  --prompt-batch N       the prompt's tokens a call, as strata-poly's\n"
    "  --kv fp16|fp8          the attention caches' format, as strata-poly's\n"
    "  --threads N            the CPU experts' threads\n"
    "  --no-graphs            no recorded launches\n"
    "  --no-gpu-experts       every routed expert on the CPU\n"
    "  --vram-reserve-mib N   VRAM left free after the expert cache is filled, as strata-poly's\n"
    "  --expert-profile FILE  the expert cache's order, as strata-poly's (read only: this tool never writes it)\n";

std::vector<int32_t> ids(const std::string& s) {
    std::vector<int32_t> out;
    for (const char* p = s.c_str(); *p;) {
        char* end = nullptr;
        out.push_back((int32_t) std::strtol(p, &end, 10));
        if (end == p) throw std::runtime_error("not a list of token ids: " + s.substr(0, 80));
        p = *end == ',' ? end + 1 : end;
    }
    return out;
}

// ids apart by commas, white space or brackets
std::vector<int32_t> ids_in_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string t = ss.str(), u;
    for (char& c : t)
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '[' || c == ']') c = ',';
    for (char c : t)
        if (!(c == ',' && (u.empty() || u.back() == ','))) u += c;
    while (!u.empty() && u.back() == ',') u.pop_back();
    return ids(u);
}

std::vector<int32_t> ids_arg(const std::string& s) { return !s.empty() && s[0] == '@' ? ids_in_file(s.substr(1)) : ids(s); }

int likeliest(const float* l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (l[i] > l[best]) best = i;
    return best;
}

void top(const std::string& what, const float* l, int n) {
    std::vector<int> o((size_t) n);
    std::iota(o.begin(), o.end(), 0);
    const int k = std::min(5, n);
    std::partial_sort(o.begin(), o.begin() + k, o.end(), [l](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
    std::printf("%s:", what.c_str());
    for (int i = 0; i < k; ++i) std::printf(" %d(%.4f)", o[(size_t) i], l[o[(size_t) i]]);
    std::printf("\n");
}

FILE* create(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    // strata-poly's default (main.cpp): the cached experts' warp kernel, the same bits
    if (!std::getenv("STRATA_SLOTS_WARP")) {
#ifdef _WIN32
        _putenv_s("STRATA_SLOTS_WARP", "1");
#else
        setenv("STRATA_SLOTS_WARP", "1", 0);
#endif
    }
#endif
    try {
        if (argc < 2) throw std::runtime_error(kUsage);
        strata::poly::Options o;
        o.model = argv[1];
        o.max_context = 4096;
        std::vector<int32_t> prompt, decode;
        std::string out = ".";
        int vocab = 0, greedy = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> const char* {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--tokens") prompt = ids_arg(val());
            else if (a == "--tokens-file") prompt = ids_in_file(val());
            else if (a == "--decode") decode = ids_arg(val());
            else if (a == "--greedy") greedy = std::atoi(val());
            else if (a == "--out") out = val();
            else if (a == "--vocab") vocab = std::atoi(val());
            else if (a == "--max-context") o.max_context = std::atoi(val());
            else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
            else if (a == "--kv") {
                const std::string f = val();
                if (f == "fp16") o.kv_format = strata::poly::KvFmt::kFp16;
                else if (f == "fp8") o.kv_format = strata::poly::KvFmt::kFp8;
                else throw std::runtime_error("--kv is fp16 or fp8");
            } else if (a == "--threads") o.threads = std::atoi(val());
            else if (a == "--no-graphs") o.graphs = false;
            else if (a == "--no-gpu-experts") o.gpu_experts = false;
            else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(val());
            else if (a == "--expert-profile") o.profile = val();
            else throw std::runtime_error("unknown option " + a + "\n" + kUsage);
        }
        if (prompt.empty()) throw std::runtime_error(std::string("no prompt tokens\n") + kUsage);
        if ((int64_t) (prompt.size() + decode.size()) + greedy > o.max_context)
            throw std::runtime_error("the prompt and the decode tokens exceed --max-context");

        auto e = strata::poly::Engine::open(o);
        std::fprintf(stderr, "loaded: %lld MiB VRAM free, %d of %d experts cached (%.2f GiB)\n",
                     (long long) e->vram_free_mib(), e->cached_experts(), e->total_experts(), e->cached_gib());
        const int nv = e->n_vocab(), nw = vocab > 0 ? std::min(vocab, nv) : nv;
        for (size_t i = 0; i + 1 < prompt.size();) {
            const int m = (int) std::min(prompt.size() - 1 - i, (size_t) e->max_read());
            e->read(prompt.data() + i, m);
            i += (size_t) m;
        }
        e->end_read();
        e->eval(&prompt.back(), 1, true);
        FILE* f = create(out + "/prompt_logits.f32");
        std::fwrite(e->logits(), sizeof(float), (size_t) nw, f);
        std::fclose(f);
        top("prompt", e->logits(), nw);
        f = create(out + "/decode_logits.f32");
        std::vector<int32_t> fed;
        int32_t next = likeliest(e->logits(), nw);
        for (size_t k = 0; k < decode.size() + (size_t) greedy; ++k) {
            const int32_t t = k < decode.size() ? decode[k] : next;
            fed.push_back(t);
            e->eval(&t, 1, true);
            std::fwrite(e->logits(), sizeof(float), (size_t) nw, f);
            top("decode " + std::to_string(k), e->logits(), nw);
            next = likeliest(e->logits(), nw);
        }
        std::fclose(f);
        std::printf("fed:");
        for (const int32_t t : fed) std::printf(" %d", t);
        std::printf("\n");
        const strata::poly::Stats& st = e->stats();
        std::fprintf(stderr, "experts from VRAM %lld (%.1f%%), CPU %lld; %d of %d experts cached (%.2f GiB)\n",
                     (long long) st.gpu_experts,
                     100.0 * (double) st.gpu_experts / (double) std::max<int64_t>(1, st.gpu_experts + st.cpu_experts),
                     (long long) st.cpu_experts, e->cached_experts(), e->total_experts(), e->cached_gib());
        return 0;
    } catch (const std::exception& x) {
        std::fprintf(stderr, "poly_logits_dump: %s\n", x.what());
        return 1;
    }
}
