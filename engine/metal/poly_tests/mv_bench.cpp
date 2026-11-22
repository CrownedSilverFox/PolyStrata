// metal/poly_tests/mv_bench.cpp - one token's quantized products (poly_mul) on the shapes of a decode step: the experts
// of a token (8 of them, by expert ids) and a dense projection, for every type poly_mul has.  Not a test: run by hand.
//   poly_mv_bench [reps (at least; enough for 20 ms a column), default 200 [types "8,22,19,..." [shapes inkling|qwen|sweep,
//                 default inkling and qwen [rounds, default 5]]]]
// inkling: Inkling 2 Mini (2,304 wide, experts of 2,304 x 1,152 with their gate and up in one call, downs of 1,152);
// qwen: Qwen3.6-35B-A3B (2,048 wide, experts of 2,048 x 512).  Each call reads other experts or another copy of the
// matrix than the last, so the weights come from memory.  Columns: "now" the engine's launch (native_mmvq_ids /
// native_mmvq); with MV_COMPARE=1 also the kernel on 1 and on 4 outputs a threadgroup ("_1" and ""), and the same
// from the type's block decoder ("deq", the dot every type had before).  Every round runs every column in turn and each
// column keeps its best round, so a Mac that is shared hurts them alike.  MV_BENCH_CACHE=dir keeps the quantized
// weights there between runs.
#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "mmq.hpp"
#include "../src/poly/launch.hpp"
#include "../src/poly/quantized.hpp"
#include "ggml.h"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
namespace k=strata::kernels;
namespace p=strata::poly;
namespace md=strata::poly::metal_detail;
int main(int argc,char** argv){
    const int reps0=argc>1?std::atoi(argv[1]):200;
    constexpr int kMaxReps=5000;
    std::vector<int> types={8,22,19,42,16,14,12,40,2,3,6,7,10,11,13,17,18,20,21,23,29,39};
    if(argc>2){types.clear();std::stringstream ss(argv[2]);std::string t;while(std::getline(ss,t,','))if(!t.empty())types.push_back(std::stoi(t));}
    const std::string only=argc>3?argv[3]:"";
    const int rounds=argc>4?std::atoi(argv[4]):5;
    const char* cache=std::getenv("MV_BENCH_CACHE");
    const bool compare=std::getenv("MV_COMPARE")!=nullptr;
    struct Col{const char* head;const char* suffix;int nr;};
    std::vector<Col> cols={{"now","",0}};
    if(compare)cols.insert(cols.end(),{{"dot 1","_1",1},{"dot 4","",4},{"deq 1","_deq1",1},{"deq 4","_deq4",4}});
    const std::string source="#include <metal_stdlib>\nusing namespace metal;\n#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n"+
                             md::read_source(STRATA_POLY_QUANT_COMMON)+md::read_source(STRATA_POLY_DEQUANT_SOURCE)+"#define MV_DEQ\n"+
                             md::read_source(STRATA_POLY_QUANT_SOURCE);
    g::Stream s;ck(g::stream_create(&s));g::metal::set_optimize(true);
    struct Shape{const char* model;const char* name;int ni,no,used,copies;};   // used 0: a dense matrix
    const Shape ink[]={{"inkling","experts gate|up",2304,2304,8,64},{"inkling","experts down",1152,2304,8,64},
                       {"inkling","dense 2304->4096",2304,4096,0,32},{"inkling","dense 4096->2304",4096,2304,0,32}};
    const Shape qwen[]={{"qwen","experts gate|up",2048,1024,8,128},{"qwen","experts down",512,2048,8,128},
                        {"qwen","dense 2048->4096",2048,4096,0,32}};
    std::vector<Shape> shapes;
    if(only=="sweep")shapes={ink[0],ink[1],ink[2],qwen[1]};
    if(only.empty()||only=="inkling")shapes.insert(shapes.end(),std::begin(ink),std::end(ink));
    if(only.empty()||only=="qwen")shapes.insert(shapes.end(),std::begin(qwen),std::end(qwen));
    std::printf("%-8s %-18s %-8s","model","shape","type");
    for(const Col& c:cols)std::printf(" %8s",c.head);
    std::printf("  GB/s now (us)\n");
    for(const Shape& sh:shapes)for(int type:types){
        if(!k::native_mmvq_supported(type) || sh.ni%int(ggml_blck_size(ggml_type(type)))) continue;
        const size_t per=k::native_mmvq_weight_bytes(type,sh.ni,sh.no);
        std::vector<unsigned char> one(per);std::mt19937 rng(1);
        const std::string file=cache?std::string(cache)+"/"+std::to_string(type)+"_"+std::to_string(sh.ni)+"_"+std::to_string(sh.no):"";
        FILE* f=cache?std::fopen(file.c_str(),"rb"):nullptr;
        if(f && std::fread(one.data(),1,per,f)==per) std::fclose(f);
        else {
            if(f) std::fclose(f);
            if(type==40){for(size_t i=0;i<per;++i)one[i]=uint8_t(i%36<4?0x30+rng()%8:rng());}
            else{auto t=ggml_type(type);ggml_quantize_init(t);std::vector<float> w(size_t(sh.no)*sh.ni);for(size_t i=0;i<w.size();++i)w[i]=std::sin(i*.37f);
                 std::vector<float> imp(sh.ni,1);ggml_quantize_chunk(t,w.data(),one.data(),0,sh.no,sh.ni,imp.data());}
            if(cache && (f=std::fopen(file.c_str(),"wb"))){std::fwrite(one.data(),1,per,f);std::fclose(f);}
        }
        Buffer dw(per*sh.copies);for(int e=0;e<sh.copies;++e)ck(g::memcpy_sync(static_cast<unsigned char*>(dw.ptr)+per*e,one.data(),per,g::kHostToDevice));
        std::vector<const void*> table(sh.copies);for(int e=0;e<sh.copies;++e)table[e]=static_cast<unsigned char*>(dw.ptr)+per*e;
        const int used=std::max(sh.used,1);
        Buffer tab(sh.copies*sizeof(void*)),di(size_t(kMaxReps+1)*used*4),dx(values(sh.ni)),q(k::native_q8_1_bytes(sh.ni,1)),y(used*sh.no*4);
        tab.put(table.data(),sh.copies*sizeof(void*));
        k::native_quantize_q8_1(dx.as(),q.ptr,sh.ni,1,s);
        std::vector<int32_t> ids(size_t(kMaxReps+1)*used);for(size_t r=0;r<ids.size();++r)ids[r]=int((r*13+r/used*5)%sh.copies);
        di.put(ids.data(),ids.size()*4);
        auto call=[&](const Col& c,int i){
            if(!c.nr){
                if(sh.used){k::MmvqIds a{tab.as<const void*>(),di.as<int32_t>()+size_t(i)*used,0,q.ptr,0,used,y.as(),size_t(sh.no)};k::native_mmvq_ids(type,a,used,sh.ni,sh.no,s);}
                else k::native_mmvq(type,table[ids[size_t(i)*used]],q.ptr,y.as(),sh.ni,sh.no,1,s);
                return;
            }
            md::QArgs a;a.p[1]=q.ptr;a.p[2]=y.ptr;a.ni=sh.ni;a.no=sh.no;a.rows=used;a.xstride=k::native_q8_1_bytes(sh.ni);a.ystride=sh.no;
            if(sh.used){a.p[0]=tab.ptr;a.p[3]=di.as<int32_t>()+size_t(i)*used;a.every=used;a.kind=1;}
            else a.p[0]=table[ids[size_t(i)*used]];
            const std::string name="poly_mul_"+std::string(md::quant_name(type))+c.suffix;
            md::dispatch("poly_mv_bench",source,name.c_str(),a,s,{uint32_t((sh.no+c.nr-1)/c.nr),uint32_t(used),1},32,
                         {a.p,6,true,sh.used?a.p[0]:nullptr});
        };
        // enough calls for 20 ms a column: a short burst left the GPU's clocks low and the times twice as far apart
        int reps=reps0;
        {
            call(cols[0],0);ck(g::stream_synchronize(s));
            const auto t0=std::chrono::steady_clock::now();
            for(int i=1;i<=reps;++i)call(cols[0],i);
            ck(g::stream_synchronize(s));
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
            if(us<20000)reps=std::min(kMaxReps,int(reps*20000/us)+1);
        }
        std::vector<double> best(cols.size(),1e30);
        std::vector<float> first;std::string wrong;   // each column's first call against the engine's: the same sums
        for(int round=0;round<rounds;++round)for(size_t v=0;v<cols.size();++v){
            call(cols[v],0);ck(g::stream_synchronize(s));
            if(!round){
                auto out=y.read(size_t(used)*sh.no);
                if(!v)first=out;
                else{float top=0,d=0;for(size_t i=0;i<out.size();++i){top=std::max(top,std::fabs(first[i]));d=std::max(d,std::fabs(out[i]-first[i]));}
                     if(!(d<=1e-4f*top))wrong+=std::string(" WRONG ")+cols[v].head;}
            }
            const auto t0=std::chrono::steady_clock::now();
            for(int i=1;i<=reps;++i)call(cols[v],i);
            ck(g::stream_synchronize(s));
            best[v]=std::min(best[v],std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/reps);
        }
        std::printf("%-8s %-18s %-8s",sh.model,sh.name,ggml_type_name(ggml_type(type)));
        for(double b:best)std::printf(" %8.1f",b);
        std::printf("  %7.1f%s\n",double(per)*used/best[0]/1e3,wrong.c_str());
        std::fflush(stdout);
    }
    ck(g::stream_destroy(s));ggml_quantize_free();return 0;
}
