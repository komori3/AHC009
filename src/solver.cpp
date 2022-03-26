#define _CRT_NONSTDC_NO_WARNINGS
#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#define ENABLE_VIS
#define ENABLE_DUMP
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#ifdef ENABLE_VIS
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
    std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

using std::vector, std::string;
using std::cin, std::cout, std::cerr, std::endl;
using ll = long long;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;



constexpr int N = 20;
constexpr int L = 200;
constexpr int inf = INT_MAX / 8;
constexpr int di[] = { 0, -1, 0, 1 };
constexpr int dj[] = { 1, 0, -1, 0 };
const string d2c = "RULD";
int c2d[256];


struct TestCase {
    int si, sj, ti, tj;
    double p;
    int h[N][N - 1];
    int v[N - 1][N];
    TestCase(std::istream& in) {
        vector<string> H(N), V(N - 1);
        in >> si >> sj >> ti >> tj >> p >> H >> V;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N - 1; j++) {
                h[i][j] = H[i][j] - '0';
            }
        }
        for (int i = 0; i < N - 1; i++) {
            for (int j = 0; j < N; j++) {
                v[i][j] = V[i][j] - '0';
            }
        }
    }
    inline bool can_move(int i, int j, int d) const {
        if (d == 0) return j < N - 1 && !h[i][j]; // R
        if (d == 1) return i > 0 && !v[i - 1][j];
        if (d == 2) return j > 0 && !h[i][j - 1];
        return i < N - 1 && !v[i][j];
    }
    string stringify() const {
        std::ostringstream oss;
        oss << format("%d %d %d %d %f\n", si, sj, ti, tj, p);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N - 1; j++) {
                oss << h[i][j];
            }
            oss << '\n';
        }
        for (int i = 0; i < N - 1; i++) {
            for (int j = 0; j < N; j++) {
                oss << v[i][j];
            }
            oss << '\n';
        }
        return oss.str();
    }
};

struct Solver {
    TestCase tc;
    Solver(const TestCase& tc) : tc(tc) {}
    string solve() {
        // Skate みたいな感じで"滑る"移動をする
    // "滑る"移動のみで到達できない場合はなるべく近くまで移動して、複数回に分けて確率を流すイメージ？

        auto [si, sj, ti, tj, p, H, V] = tc;        

        int dist_straight[4][N][N];
        Fill(dist_straight, -1);

        // 右にいくつ進めるか？を前計算
        {
            auto& rdist = dist_straight[0];
            for (int i = 0; i < N; i++) {
                rdist[i][N - 1] = 0;
                int d = 0;
                for (int j = N - 2; j >= 0; j--) {
                    if (H[i][j] || (i == ti && j == tj)) d = -1;
                    d++;
                    rdist[i][j] = d;
                }
            }
        }
        {
            auto& udist = dist_straight[1];
            for (int j = 0; j < N; j++) {
                udist[0][j] = 0;
                int d = 0;
                for (int i = 1; i <= N - 1; i++) {
                    if (V[i - 1][j] || (i == ti && j == tj)) d = -1;
                    d++;
                    udist[i][j] = d;
                }
            }
        }
        {
            auto& ldist = dist_straight[2];
            for (int i = 0; i < N; i++) {
                ldist[i][0] = 0;
                int d = 0;
                for (int j = 1; j <= N - 1; j++) {
                    if (H[i][j - 1] || (i == ti && j == tj)) d = -1;
                    d++;
                    ldist[i][j] = d;
                }
            }
        }
        {
            auto& ddist = dist_straight[3];
            for (int j = 0; j < N; j++) {
                ddist[N - 1][j] = 0;
                int d = 0;
                for (int i = N - 2; i >= 0; i--) {
                    if (V[i][j] || (i == ti && j == tj)) d = -1;
                    d++;
                    ddist[i][j] = d;
                }
            }
        }

        // grid 上の dijkstra
        using Edge = std::tuple<int, int, int>;
        using PQ = std::priority_queue<Edge, vector<Edge>, std::greater<Edge>>;

        int dist[N][N];
        pii prev[N][N]; // recon
        Fill(dist, inf); Fill(prev, pii(-1, -1));

        PQ pq;
        pq.emplace(0, si, sj);
        dist[si][sj] = 0;
        while (!pq.empty()) {
            auto [mincost, i, j] = pq.top(); pq.pop();
            if (dist[i][j] < mincost) continue;
            // 障害物にぶつかる or ゴールに到達するまで直進
            for (int d = 0; d < 4; d++) {
                int move_len = dist_straight[d][i][j];
                if (!move_len) continue;
                int ni = i + di[d] * move_len, nj = j + dj[d] * move_len;
                if (chmin(dist[ni][nj], dist[i][j] + move_len)) {
                    pq.emplace(dist[ni][nj], ni, nj);
                    prev[ni][nj] = pii(d, move_len);
                }
            }
        }

        string path;
        int i = ti, j = tj;
        while (true) {
            auto [d, len] = prev[i][j];
            if (d == -1) break;
            path += string(len + 7, d2c[d]);
            i -= di[d] * len;
            j -= dj[d] * len;
        }
        reverse(path.begin(), path.end());
        return path.size() > 200 ? "" : path;
    }
};

int compute_score(const TestCase& tc, const string& ans) {
    if (ans.size() > 200) {
        dump("too long");
        return -1;
    }
    auto crt = make_vector(0.0, N, N);
    crt[tc.si][tc.sj] = 1.0;
    double sum = 0.0, goal = 0.0;
    for (int t = 0; t < ans.size(); t++) {
        auto next = make_vector(0.0, N, N);
        int d = c2d[ans[t]];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (crt[i][j] > 0.0) {
                    if (tc.can_move(i, j, d)) {
                        int i2 = i + di[d];
                        int j2 = j + dj[d];
                        next[i2][j2] += crt[i][j] * (1.0 - tc.p);
                        next[i][j] += crt[i][j] * tc.p;
                    }
                    else {
                        next[i][j] += crt[i][j];
                    }
                }
            }
        }
        crt = next;
        sum += crt[tc.ti][tc.tj] * (2 * L - t);
        goal += crt[tc.ti][tc.tj];
        crt[tc.ti][tc.tj] = 0.0;
    }
    crt[tc.ti][tc.tj] = goal;
    return (int)round((1e8 * sum / (2 * L)));
}

void batch_test() {

    ll total = 0;

    for (int seed = 0; seed < 100; seed++) {
        std::ifstream ifs(format("tools/in/%04d.txt", seed));
        std::istream& in = ifs;
        std::ofstream ofs(format("tools/out/%04d.txt", seed));
        std::ostream& out = ofs;

        TestCase tc(in);

        Solver solver(tc);
        auto ans = solver.solve();

        total += compute_score(tc, ans);

        dump(seed, compute_score(tc, ans));

        out << ans << endl;

        ifs.close();
        ofs.close();
    }

    cout << total << endl;
}

int main(int argc, char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    c2d['R'] = 0; c2d['U'] = 1; c2d['L'] = 2; c2d['D'] = 3;

#ifdef _MSC_VER
    batch_test();
#else
    std::istream& in = cin;
    std::ostream& out = cout;

    TestCase tc(in);

    Solver solver(tc);
    auto ans = solver.solve();

    dump(ans);

    out << ans << endl;

#endif

    return 0;
}