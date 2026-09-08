#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <poll.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t Slot = 8 * MiB;
struct Desc { uint64_t src, dst, length; };
struct Message { uint64_t kind, generation, value; };
enum : uint64_t { SOURCE_READY=1, CREDIT, DONE, ACK, STALE, REJECTED, CLOSED };
using Launch = void (*)(uint32_t, void*, void*, void*, void*, uint64_t, uint32_t);

void check(aclError rc, const char* operation) {
    if (rc != ACL_SUCCESS) throw std::runtime_error(std::string(operation) + " rc=" + std::to_string(rc));
}
#define ACL(expr) check((expr), #expr)
void io(int fd, void* address, size_t n, bool send) {
    auto* p = static_cast<char*>(address);
    auto deadline = Clock::now() + std::chrono::seconds(90);
    while (n) {
        int remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
        if (remaining <= 0) throw std::runtime_error("control channel timeout");
        pollfd f{fd, static_cast<short>(send ? POLLOUT : POLLIN), 0};
        int rc = poll(&f, 1, remaining);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("control poll timeout/error");
        ssize_t bytes = send ? ::send(fd, p, n, MSG_NOSIGNAL) : ::recv(fd, p, n, 0);
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) throw std::runtime_error("control peer exited/error");
        p += bytes; n -= bytes;
    }
}
template<class T> void send_value(int fd, T v) { io(fd, &v, sizeof(v), true); }
template<class T> T recv_value(int fd) { T v{}; io(fd, &v, sizeof(v), false); return v; }
void send_message(int fd, uint64_t kind, uint64_t generation, uint64_t value=0) {
    send_value(fd, Message{kind, generation, value});
}
Message expect(int fd, uint64_t kind, uint64_t generation) {
    auto m = recv_value<Message>(fd);
    if (m.kind != kind || m.generation != generation)
        throw std::runtime_error("unexpected control message/generation");
    return m;
}
void record(const std::string& s) {
    std::string line=s+"\n"; size_t offset=0;
    while(offset<line.size()) {
        auto n=::write(STDOUT_FILENO,line.data()+offset,line.size()-offset);
        if(n<0&&errno==EINTR) continue;
        if(n<=0) throw std::runtime_error("result write failed");
        offset+=n;
    }
}

struct Device {
    int id;
    aclrtStream stream{};
    void* lib{};
    Launch launch{};
    Device(int n): id(n) {
        ACL(aclInit(nullptr)); ACL(aclrtSetDevice(id)); ACL(aclrtCreateStream(&stream));
        lib=dlopen("./build-release/lib/libkv_copy.so", RTLD_NOW|RTLD_LOCAL);
        if (!lib) throw std::runtime_error(dlerror());
        launch=reinterpret_cast<Launch>(dlsym(lib,"kv_copy_launch"));
        if (!launch) throw std::runtime_error(dlerror());
    }
    void sync() { ACL(aclrtSynchronizeStream(stream)); }
    void close() {
        sync(); ACL(aclrtDestroyStream(stream)); stream=nullptr;
        dlclose(lib); lib=nullptr; ACL(aclrtResetDevice(id)); ACL(aclFinalize());
    }
};
struct Allocation {
    void* p{}; uint64_t size;
    explicit Allocation(uint64_t n): size(n) { ACL(aclrtMalloc(&p, n, ACL_MEM_MALLOC_HUGE_ONLY_P2P)); }
    void release() { if(p) { ACL(aclrtFree(p)); p=nullptr; } }
    ~Allocation() { if(p) aclrtFree(p); }
};
struct Mapping {
    std::array<char,65> key{};
    void* p{};
    bool open=false;
    void export_memory(Allocation& allocation, const std::vector<int32_t>& pids) {
        ACL(aclrtIpcMemGetExportKey(allocation.p,allocation.size,key.data(),key.size(),0)); open=true;
        auto allowed=pids; ACL(aclrtIpcMemSetImportPid(key.data(),allowed.data(),allowed.size()));
    }
    void import_memory(std::array<char,65> k) {
        key=k; ACL(aclrtIpcMemImportByKey(&p,key.data(),ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS)); open=true;
    }
    void close() { if(open) { ACL(aclrtIpcMemClose(key.data())); open=false; p=nullptr; } }
    ~Mapping() { if(open) aclrtIpcMemClose(key.data()); }
};
void validate(const std::vector<Desc>& plan,uint64_t src_capacity,uint64_t dst_capacity) {
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    for(auto d:plan) {
        if(d.length==0) continue;
        if(d.src>src_capacity || d.length>src_capacity-d.src ||
           d.dst>dst_capacity || d.length>dst_capacity-d.dst) throw std::runtime_error("descriptor out of bounds");
        ranges.emplace_back(d.dst,d.dst+d.length);
    }
    std::sort(ranges.begin(),ranges.end());
    for(size_t i=1;i<ranges.size();++i) if(ranges[i].first<ranges[i-1].second)
        throw std::runtime_error("overlapping destination descriptors");
}
std::vector<Desc> tiles(uint64_t bytes,uint64_t src=0,uint64_t dst=0) {
    std::vector<Desc> p;
    for(uint64_t i=0;i<bytes;i+=65536) p.push_back({src+i,dst+i,std::min<uint64_t>(65536,bytes-i)});
    return p;
}
struct Plan {
    Allocation memory{2*MiB};
    std::vector<Desc> host;
    void set(std::vector<Desc> p,uint64_t sc,uint64_t dc) {
        validate(p,sc,dc); host=std::move(p);
        if(host.size()*sizeof(Desc)>memory.size) throw std::runtime_error("descriptor capacity exceeded");
        if(!host.empty()) ACL(aclrtMemcpy(memory.p,memory.size,host.data(),host.size()*sizeof(Desc),ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void run(Device& d,void* src,void* dst,uint32_t cores,uint32_t no_l2) {
        if(!host.empty()) d.launch(cores,d.stream,src,dst,memory.p,host.size(),no_l2);
    }
    uint64_t active(uint32_t cores) const {
        std::set<uint64_t> workers;
        for(size_t i=0;i<host.size();++i) if(host[i].length) workers.insert(i%cores);
        return workers.size();
    }
};
std::vector<uint8_t> pattern(uint64_t size,uint64_t gen) {
    std::vector<uint8_t> b(size);
    for(uint64_t i=0;i<size;++i) b[i]=(17*i+13*(i>>8)+29*(i>>16)+37*gen)%251;
    return b;
}
void h2d(Allocation& a,const std::vector<uint8_t>& b) {
    ACL(aclrtMemcpy(a.p,a.size,b.data(),b.size(),ACL_MEMCPY_HOST_TO_DEVICE));
}
void verify(void* p,const std::vector<uint8_t>& expected,const char* stage) {
    std::vector<uint8_t> actual(expected.size());
    ACL(aclrtMemcpy(actual.data(),actual.size(),p,actual.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    auto it=std::mismatch(expected.begin(),expected.end(),actual.begin());
    if(it.first!=expected.end()) throw std::runtime_error(std::string(stage)+" mismatch offset="+
        std::to_string(it.first-expected.begin())+" expected="+std::to_string(*it.first)+" actual="+std::to_string(*it.second));
}
std::vector<Desc> correctness_plan(int test,uint64_t slot) {
    if(test==0) return {};
    if(test==1) return {{UINT64_MAX,UINT64_MAX,0}};
    std::vector<Desc> p;
    std::set<int> used;
    for(int page:{7,1,17,2,7,30}) if(used.insert(page).second) {
        p.push_back({128+uint64_t(page)*131072,slot*Slot+256+uint64_t(page)*131072,65632});
        p.push_back({5*MiB+64+uint64_t(page)*8192,slot*Slot+5*MiB+160+uint64_t(page)*8192,4160});
        p.push_back({6*MiB+96+uint64_t(page)*256,slot*Slot+6*MiB+128+uint64_t(page)*256,96});
    }
    p.push_back({UINT64_MAX,UINT64_MAX,0});
    if(test==3) {
        p.push_back({7*MiB+17,slot*Slot+7*MiB+43,31});
        p.push_back({7*MiB+4099,slot*Slot+7*MiB+8197,65543});
    }
    return p;
}

struct Options { std::string phase="p1",mode="pull"; std::vector<int> devices{0,2}; int cores=32,no_l2=0; };
Options options(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i]; if(i+1>=argc) throw std::runtime_error("missing argument"); std::string v=argv[++i];
        if(k=="--phase") o.phase=v; else if(k=="--mode") o.mode=v;
        else if(k=="--cores") o.cores=std::stoi(v); else if(k=="--no-l2") o.no_l2=std::stoi(v);
        else if(k=="--devices") { o.devices.clear(); std::istringstream s(v); std::string n; while(std::getline(s,n,',')) o.devices.push_back(std::stoi(n)); }
        else throw std::runtime_error("unknown argument");
    }
    if((o.phase!="p1"&&o.phase!="p2")||(o.mode!="pull"&&o.mode!="push")||o.cores<1||o.cores>32||
        o.devices.size()<2||std::set<int>(o.devices.begin(),o.devices.end()).size()!=o.devices.size()) throw std::runtime_error("invalid options");
    if(o.phase=="p2"&&o.mode!="pull") throw std::runtime_error("P2 push is not implemented");
    return o;
}
void p1(Device& d,const Options& o,int rank,const std::vector<int>& channels,
    Allocation& src,Allocation& dst,void* remote_source,const std::vector<void*>& remote_destinations) {
    Allocation output(2*Slot);
    Plan copy,consume;
    consume.set(tiles(2*Slot),2*Slot,2*Slot);
    uint64_t gen=0;
    for(int test=0;test<4;++test) {
        std::vector<uint8_t> expected(2*Slot,0xCC);
        h2d(dst,expected);
        for(int epoch=0;epoch<4;++epoch) {
            ++gen; uint64_t slot=epoch%2;
            auto input=pattern(Slot,gen);
            copy.set(correctness_plan(test,slot),Slot,2*Slot);
            for(auto x:copy.host) if(x.length) std::copy_n(input.begin()+x.src,x.length,expected.begin()+x.dst);
            if(rank==0) {
                h2d(src,input); d.sync();
                for(auto fd:channels) {
                    send_message(fd,STALE,gen-1); expect(fd,REJECTED,gen-1);
                    send_message(fd,SOURCE_READY,gen);
                }
                for(size_t i=0;i<channels.size();++i) {
                    expect(channels[i],CREDIT,gen);
                    if(o.mode=="push") { copy.run(d,src.p,remote_destinations[i],o.cores,o.no_l2); d.sync(); send_message(channels[i],DONE,gen); }
                }
                for(auto fd:channels) expect(fd,ACK,gen);
            } else {
                auto stale=recv_value<Message>(channels[0]);
                if(stale.kind!=STALE||stale.generation>=gen) throw std::runtime_error("stale generation injection invalid");
                send_message(channels[0],REJECTED,stale.generation);
                expect(channels[0],SOURCE_READY,gen);
                // Prewarm destination cache and complete its prior NPU use before credit.
                consume.run(d,dst.p,output.p,o.cores,0); d.sync();
                if(epoch==1&&rank==1) {
                    poll(nullptr,0,40); // Intentional credit-delay test, never a visibility workaround.
                    pollfd f{channels[0],POLLIN,0};
                    if(poll(&f,1,0)!=0) throw std::runtime_error("owner sent completion before credit");
                }
                send_message(channels[0],CREDIT,gen);
                if(o.mode=="pull") { copy.run(d,remote_source,dst.p,o.cores,o.no_l2); d.sync(); }
                else expect(channels[0],DONE,gen);
                consume.run(d,dst.p,output.p,o.cores,0); d.sync();
                verify(output.p,expected,"local NPU consumption"); verify(dst.p,expected,"scratch canary");
                send_message(channels[0],ACK,gen);
                record("{\"phase\":\"p1\",\"status\":\"pass\",\"mode\":\""+o.mode+"\",\"rank\":"+std::to_string(rank)+
                    ",\"device\":"+std::to_string(d.id)+",\"generation\":"+std::to_string(gen)+",\"slot\":"+std::to_string(slot)+
                    ",\"test\":"+std::to_string(test)+",\"descriptors\":"+std::to_string(copy.host.size())+
                    ",\"launch_cores\":"+std::to_string(o.cores)+",\"active_cores\":"+std::to_string(copy.active(o.cores))+
                    ",\"no_l2\":"+std::to_string(o.no_l2)+",\"validated_bytes\":16777216,\"npu_consume\":true,\"stale_rejected\":true}");
            }
        }
    }
    d.sync(); output.release(); copy.memory.release(); consume.memory.release();
}

std::pair<double,double> timed(Device& d,Plan& plan,void* src,void* dst,int cores,int no_l2,aclrtEvent begin,aclrtEvent end) {
    auto start=Clock::now(); ACL(aclrtRecordEvent(begin,d.stream)); plan.run(d,src,dst,cores,no_l2);
    ACL(aclrtRecordEvent(end,d.stream)); ACL(aclrtSynchronizeEvent(end));
    double wall=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    float ms; ACL(aclrtEventElapsedTime(&ms,begin,end)); return {ms*1000.0,wall};
}
void p2(Device& d,const Options& o,int rank,const std::vector<int>& channels,Allocation& src,Allocation& dst,void* remote_source) {
    auto input=pattern(src.size,1); h2d(src,input); d.sync();
    if(rank==0) {
        for(auto fd:channels) send_message(fd,SOURCE_READY,1);
        for(auto fd:channels) expect(fd,ACK,1);
        return;
    }
    expect(channels[0],SOURCE_READY,1);
    Allocation output(256*MiB);
    Plan plan,consume; aclrtEvent begin,end; ACL(aclrtCreateEvent(&begin)); ACL(aclrtCreateEvent(&end));
    plan.set({},src.size,dst.size);
    for(int i=0;i<31;++i) {
        auto t=timed(d,plan,src.p,dst.p,o.cores,o.no_l2,begin,end);
        record("{\"phase\":\"p2\",\"backend\":\"empty_event\",\"rank\":"+std::to_string(rank)+",\"device_us\":"+std::to_string(t.first)+",\"host_us\":"+std::to_string(t.second)+"}");
    }
    for(uint64_t bytes:{4096ULL,65536ULL,1ULL*MiB,16ULL*MiB,64ULL*MiB,256ULL*MiB}) {
        for(int rotating=0;rotating<2;++rotating) {
            // All descriptors and allocations are prepared outside the timed interval.
            for(int i=-10;i<31;++i) {
                uint64_t offset=rotating ? uint64_t(i+10)%2*bytes : 0;
                plan.set(tiles(bytes,offset,0),src.size,dst.size);
                for(int order=0;order<2;++order) {
                    bool remote=((i+10+order)%2)==1;
                    auto t=timed(d,plan,remote?remote_source:src.p,dst.p,o.cores,o.no_l2,begin,end);
                    if(i>=0) record("{\"phase\":\"p2\",\"backend\":\""+std::string(remote?"ipc_pull":"local")+
                        "\",\"rank\":"+std::to_string(rank)+",\"owner\":"+std::to_string(o.devices[0])+",\"device\":"+std::to_string(d.id)+
                        ",\"payload\":"+std::to_string(bytes)+",\"iteration\":"+std::to_string(i)+",\"launch_cores\":"+std::to_string(o.cores)+
                        ",\"active_cores\":"+std::to_string(plan.active(o.cores))+",\"no_l2\":"+std::to_string(o.no_l2)+
                        ",\"rotating\":"+std::to_string(rotating)+",\"arena_bytes\":"+std::to_string(rotating?2*bytes:bytes)+
                        ",\"device_us\":"+std::to_string(t.first)+",\"host_us\":"+std::to_string(t.second)+"}");
                }
            }
            // Validate every arena view outside timing with a local NPU consumer.
            for(int view=0;view<(rotating?2:1);++view) {
                uint64_t offset=view*bytes;
                plan.set(tiles(bytes,offset,0),src.size,dst.size); plan.run(d,remote_source,dst.p,o.cores,o.no_l2); d.sync();
                consume.set(tiles(bytes),dst.size,output.size); consume.run(d,dst.p,output.p,o.cores,0); d.sync();
                std::vector<uint8_t> expected(input.begin()+offset,input.begin()+offset+bytes);
                verify(output.p,expected,"P2 local NPU consumption"); verify(dst.p,expected,"P2 payload");
                record("{\"phase\":\"p2_validation\",\"status\":\"pass\",\"rank\":"+std::to_string(rank)+
                    ",\"payload\":"+std::to_string(bytes)+",\"rotating\":"+std::to_string(rotating)+",\"view\":"+std::to_string(view)+",\"npu_consume\":true}");
            }
        }
    }
    ACL(aclrtDestroyEvent(begin)); ACL(aclrtDestroyEvent(end)); plan.memory.release(); consume.memory.release(); output.release();
    send_message(channels[0],ACK,1);
}

void process(const Options& o,int rank,const std::vector<int>& channels) {
    Device d(o.devices[rank]);
    int32_t pid; ACL(aclrtDeviceGetBareTgid(&pid));
    std::vector<int32_t> peers;
    if(rank==0) {
        for(auto fd:channels) peers.push_back(recv_value<int32_t>(fd));
        for(auto fd:channels) send_value(fd,pid);
    } else { send_value(channels[0],pid); peers.push_back(recv_value<int32_t>(channels[0])); }
    if(rank!=0) {
        int32_t reachable=0; ACL(aclrtDeviceCanAccessPeer(&reachable,o.devices[rank],o.devices[0]));
        if(reachable!=1) throw std::runtime_error("peer access unsupported");
    }
    Allocation src(o.phase=="p1"?Slot:512*MiB),dst(o.phase=="p1"?2*Slot:256*MiB);
    Mapping source;
    std::vector<Mapping> destinations(channels.size());
    std::vector<void*> remote_dst;
    if(rank==0) {
        source.export_memory(src,peers);
        for(auto fd:channels) send_value(fd,source.key);
        if(o.mode=="push") for(size_t i=0;i<channels.size();++i) {
            destinations[i].import_memory(recv_value<std::array<char,65>>(channels[i])); remote_dst.push_back(destinations[i].p);
        }
    } else {
        source.import_memory(recv_value<std::array<char,65>>(channels[0]));
        if(o.mode=="push") { destinations[0].export_memory(dst,peers); send_value(channels[0],destinations[0].key); }
    }
    record("{\"phase\":\"p0\",\"status\":\"ipc_ready\",\"device\":"+std::to_string(d.id)+",\"rank\":"+std::to_string(rank)+
        ",\"bare_tgid\":"+std::to_string(pid)+",\"allocation\":\"ACL_MEM_MALLOC_HUGE_ONLY_P2P\",\"import_flags\":1,\"pid_validation\":true}");
    if(o.phase=="p1") p1(d,o,rank,channels,src,dst,source.p,remote_dst);
    else p2(d,o,rank,channels,src,dst,source.p);
    d.sync();
    // Importers close first; exporters wait for that confirmation before close/free.
    if(rank==0) {
        for(auto& m:destinations) m.close();
        for(auto fd:channels) send_message(fd,CLOSED,0);
        for(auto fd:channels) expect(fd,CLOSED,0);
        source.close();
    } else {
        source.close(); expect(channels[0],CLOSED,0); destinations[0].close(); send_message(channels[0],CLOSED,0);
    }
    src.release(); dst.release(); d.close();
    record("{\"phase\":\"cleanup\",\"status\":\"pass\",\"rank\":"+std::to_string(rank)+"}");
}
int main(int argc,char** argv) {
    std::vector<pid_t> children;
    try {
        auto o=options(argc,argv);
        std::vector<std::array<int,2>> sockets(o.devices.size()-1);
        for(auto& s:sockets) if(socketpair(AF_UNIX,SOCK_STREAM,0,s.data())) throw std::runtime_error("socketpair failed");
        for(size_t r=1;r<o.devices.size();++r) {
            pid_t pid=fork(); if(pid<0) throw std::runtime_error("fork failed");
            if(pid==0) {
                children.clear();
                for(size_t j=0;j<sockets.size();++j) { ::close(sockets[j][0]); if(j!=r-1) ::close(sockets[j][1]); }
                try { process(o,r,{sockets[r-1][1]}); _exit(0); }
                catch(const std::exception& e) { std::cerr<<"rank="<<r<<" error: "<<e.what()<<std::endl; _exit(1); }
            }
            children.push_back(pid);
        }
        std::vector<int> channels;
        for(auto& s:sockets) { ::close(s[1]); channels.push_back(s[0]); }
        process(o,0,channels);
        bool okay=true;
        for(auto pid:children) { int status; if(waitpid(pid,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status)) okay=false; }
        children.clear(); if(!okay) throw std::runtime_error("child failed");
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"owner error: "<<e.what()<<std::endl;
        for(auto pid:children) kill(pid,SIGTERM);
        for(auto pid:children) waitpid(pid,nullptr,0);
        return 1;
    }
}
