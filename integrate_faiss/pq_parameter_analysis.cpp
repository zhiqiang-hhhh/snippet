#include <faiss/IndexPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/IndexScalarQuantizer.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <unordered_set>

struct Options {
    int dim = 128;
    int nb = 20000;
    int nq = 500;
    int k = 10;

    // HNSW
    int hnsw_M = 32;
    int efC = 200;
    int efS = 64;

    // PQ (for HNSW-PQ)
    int pq_m = 16;
    int pq_nbits = 8;
    double pq_train_ratio = 1.0; // fraction of nb used for PQ training [0,1]

    // SQ (for HNSW-SQ)
    int sq_qtype = (int)faiss::ScalarQuantizer::QuantizerType::QT_8bit;

    // Which tests to run: all, hnsw_flat, hnsw_sq, hnsw_pq
    std::vector<std::string> which = {"all"};

    // Output
    std::string out_csv = "hnsw_index_types_results.csv";
};

class PQBenchmark {
private:
    int dim;
    int nb;
    int nq;
    int k;
    std::vector<float> database;
    std::vector<float> queries;
    std::vector<faiss::idx_t> ground_truth;
    
public:
    PQBenchmark(int dim = 128, int nb = 20000, int nq = 500, int k = 10) 
        : dim(dim), nb(nb), nq(nq), k(k) {
        std::cout << "=== HNSW Index Types Benchmark (C++) ===" << std::endl;
        std::cout << "配置: dim=" << dim << ", nb=" << nb << ", nq=" << nq << ", k=" << k << std::endl;
        generateData();
        computeGroundTruth();
    }
    
    void generateData() {
        std::cout << "生成测试数据..." << std::endl;
        
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0, 1.0);
        
        database.resize(nb * dim);
        queries.resize(nq * dim);
        
        // 生成数据库向量
        for (int i = 0; i < nb * dim; i++) {
            database[i] = dist(rng);
        }
        
        // 生成查询向量
        for (int i = 0; i < nq * dim; i++) {
            queries[i] = dist(rng);
        }
        
        // L2规范化
        normalizeVectors(database.data(), nb, dim);
        normalizeVectors(queries.data(), nq, dim);
    }
    
    void normalizeVectors(float* vectors, int n, int d) {
        for (int i = 0; i < n; i++) {
            float norm = 0.0;
            for (int j = 0; j < d; j++) {
                norm += vectors[i * d + j] * vectors[i * d + j];
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < d; j++) {
                    vectors[i * d + j] /= norm;
                }
            }
        }
    }
    
    void computeGroundTruth() {
        std::cout << "计算真实最近邻..." << std::endl;
        
        faiss::IndexFlatL2 index(dim);
        index.add(nb, database.data());
        
        std::vector<float> distances(nq * k);
        ground_truth.resize(nq * k);
        
        index.search(nq, queries.data(), k, distances.data(), ground_truth.data());
    }
    
    struct TestResult {
        std::string method;       // HNSW-Flat / HNSW-SQ / HNSW-PQ
        std::string params;       // printable params
        int hnsw_M = 0;
        int efC = 0;
        int efS = 0;
        int pq_m = 0;
        int pq_nbits = 0;
        int qtype = -1;           // for SQ

        double train_time = 0.0;  // seconds
        double add_time = 0.0;    // seconds
        double build_time = 0.0;  // train + add seconds
        double search_time_ms = 0.0; // per query
        double recall_1 = 0.0;
        double recall_5 = 0.0;
        double recall_10 = 0.0;
        double memory_mb = 0.0;   // encoding only
        double compression_ratio = 0.0;
    };

    // 小工具：qtype 名称
    static const char* qtypeName(int qtype) {
        using QT = faiss::ScalarQuantizer::QuantizerType;
        switch (qtype) {
            case QT::QT_8bit:          return "QT_8bit";
            case QT::QT_4bit:          return "QT_4bit";
            case QT::QT_8bit_uniform:  return "QT_8bit_uniform";
            case QT::QT_fp16:          return "QT_fp16";
#ifdef FAISS_HAVE_QT_8BIT_DIRECT
            case QT::QT_8bit_direct:   return "QT_8bit_direct";
#endif
            default:                   return "unknown";
        }
    }

    // HNSW-Flat
    TestResult testHNSWFlat(int hnsw_m, int efC, int efS) {
        std::cout << "  测试 IndexHNSWFlat (M=" << hnsw_m << ", efC=" << efC << ", efS=" << efS << ")..." << std::endl;
        TestResult r{};
        r.method = "HNSW-Flat";
        r.params = "HNSW_M=" + std::to_string(hnsw_m) + ", efC=" + std::to_string(efC) + ", efS=" + std::to_string(efS);
        r.hnsw_M = hnsw_m; r.efC = efC; r.efS = efS;
        try {
            faiss::IndexHNSWFlat index(dim, hnsw_m);
            index.hnsw.efConstruction = efC;
            // no train for HNSW-Flat
            r.train_time = 0.0;
            auto t_add0 = std::chrono::high_resolution_clock::now();
            index.add(nb, database.data());
            auto t_add1 = std::chrono::high_resolution_clock::now();
            r.add_time = std::chrono::duration<double>(t_add1 - t_add0).count();
            r.build_time = r.train_time + r.add_time;
            std::cout << std::fixed << std::setprecision(3)
                      << "    训练(train)= " << r.train_time << " s, 建索引(add)= " << r.add_time
                      << " s, 总构建= " << r.build_time << " s" << std::endl;

            index.hnsw.efSearch = efS;
            std::vector<float> distances(nq * k);
            std::vector<faiss::idx_t> labels(nq * k);
            auto t0 = std::chrono::high_resolution_clock::now();
            index.search(nq, queries.data(), k, distances.data(), labels.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            r.search_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

            r.recall_1 = computeRecall(labels, ground_truth, 1);
            r.recall_5 = computeRecall(labels, ground_truth, 5);
            r.recall_10 = computeRecall(labels, ground_truth, k);

            double code_bytes = (double)dim * 4.0; // float32 per dim
            r.memory_mb = (code_bytes * nb) / (1024.0 * 1024.0);
            r.compression_ratio = ((double)dim * 4.0) / code_bytes; // =1
        } catch (const std::exception& e) {
            std::cerr << "    错误: " << e.what() << std::endl;
            r.build_time = -1;
        }
        return r;
    }

    // HNSW-SQ
    TestResult testHNSWSQ(int qtype, int hnsw_m, int efC, int efS) {
        std::cout << "  测试 IndexHNSWSQ (" << qtypeName(qtype) << ", M=" << hnsw_m << ", efC=" << efC << ", efS=" << efS << ")..." << std::endl;
        TestResult r{};
        r.method = "HNSW-SQ";
        r.params = "HNSW_M=" + std::to_string(hnsw_m) + ", " + qtypeName(qtype) + ", efS=" + std::to_string(efS);
        r.hnsw_M = hnsw_m; r.efC = efC; r.efS = efS; r.qtype = qtype;
        try {
            faiss::IndexHNSWSQ index(dim, (faiss::ScalarQuantizer::QuantizerType)qtype, hnsw_m, faiss::METRIC_L2);
            index.hnsw.efConstruction = efC;
            auto t0 = std::chrono::high_resolution_clock::now();
            index.train(nb, database.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            r.train_time = std::chrono::duration<double>(t1 - t0).count();
            auto t2 = std::chrono::high_resolution_clock::now();
            index.add(nb, database.data());
            auto t3 = std::chrono::high_resolution_clock::now();
            r.add_time = std::chrono::duration<double>(t3 - t2).count();
            r.build_time = r.train_time + r.add_time;
            std::cout << std::fixed << std::setprecision(3)
                      << "    训练(train)= " << r.train_time << " s, 建索引(add)= " << r.add_time
                      << " s, 总构建= " << r.build_time << " s" << std::endl;

            index.hnsw.efSearch = efS;
            std::vector<float> distances(nq * k);
            std::vector<faiss::idx_t> labels(nq * k);
            t0 = std::chrono::high_resolution_clock::now();
            index.search(nq, queries.data(), k, distances.data(), labels.data());
            t1 = std::chrono::high_resolution_clock::now();
            r.search_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

            r.recall_1 = computeRecall(labels, ground_truth, 1);
            r.recall_5 = computeRecall(labels, ground_truth, 5);
            r.recall_10 = computeRecall(labels, ground_truth, k);

            int bits = 8;
            using QT = faiss::ScalarQuantizer::QuantizerType;
            switch ((QT)qtype) {
                case QT::QT_8bit:
                case QT::QT_8bit_uniform:
                    bits = 8; break;
                case QT::QT_4bit:
                    bits = 4; break;
                case QT::QT_fp16:
                    bits = 16; break;
                default: bits = 8; break;
            }
            double code_bytes = (double)dim * bits / 8.0;
            r.memory_mb = (code_bytes * nb) / (1024.0 * 1024.0);
            r.compression_ratio = ((double)dim * 4.0) / code_bytes;
        } catch (const std::exception& e) {
            std::cerr << "    错误: " << e.what() << std::endl;
            r.build_time = -1;
        }
        return r;
    }

    // HNSW-PQ
    TestResult testHNSWPQ(int pq_m, int pq_nbits, int hnsw_m, int efC, int efS, double train_ratio) {
        std::cout << "  测试 IndexHNSWPQ (M=" << hnsw_m << ", PQ(m=" << pq_m << ", nbits=" << pq_nbits << "), efC=" << efC << ", efS=" << efS << ", tr=" << train_ratio << ")..." << std::endl;
        TestResult r{};
        r.method = "HNSW-PQ";
        r.params = "HNSW_M=" + std::to_string(hnsw_m) + ", PQ(m=" + std::to_string(pq_m) + ",nbits=" + std::to_string(pq_nbits) + ")";
        r.hnsw_M = hnsw_m; r.efC = efC; r.efS = efS; r.pq_m = pq_m; r.pq_nbits = pq_nbits;
        try {
            if (dim % pq_m != 0) {
                std::cout << "    跳过：dim=" << dim << " 不能被 m=" << pq_m << " 整除" << std::endl;
                r.build_time = -1; return r;
            }
            faiss::IndexHNSWPQ index(dim, pq_m, hnsw_m, pq_nbits);
            index.hnsw.efConstruction = efC;
            // Decide training subset size based on ratio
            int train_n = (int)std::llround((double)nb * train_ratio);
            if (train_n < 1) train_n = 1;
            if (train_n > nb) train_n = nb;
            if (train_n < nb) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3) << (double)train_n / (double)nb;
                std::cout << "    使用训练样本: " << train_n << "/" << nb << " (ratio=" << oss.str() << ")" << std::endl;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            index.train(train_n, database.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            r.train_time = std::chrono::duration<double>(t1 - t0).count();
            auto t2 = std::chrono::high_resolution_clock::now();
            index.add(nb, database.data());
            auto t3 = std::chrono::high_resolution_clock::now();
            r.add_time = std::chrono::duration<double>(t3 - t2).count();
            r.build_time = r.train_time + r.add_time;
            std::cout << std::fixed << std::setprecision(3)
                      << "    训练(train)= " << r.train_time << " s, 建索引(add)= " << r.add_time
                      << " s, 总构建= " << r.build_time << " s" << std::endl;

            index.hnsw.efSearch = efS;
            std::vector<float> distances(nq * k);
            std::vector<faiss::idx_t> labels(nq * k);
            t0 = std::chrono::high_resolution_clock::now();
            index.search(nq, queries.data(), k, distances.data(), labels.data());
            t1 = std::chrono::high_resolution_clock::now();
            r.search_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

            r.recall_1 = computeRecall(labels, ground_truth, 1);
            r.recall_5 = computeRecall(labels, ground_truth, 5);
            r.recall_10 = computeRecall(labels, ground_truth, k);

            double code_bytes = (double)pq_m * pq_nbits / 8.0;
            r.memory_mb = (code_bytes * nb) / (1024.0 * 1024.0);
            r.compression_ratio = ((double)dim * 4.0) / code_bytes;
        } catch (const std::exception& e) {
            std::cerr << "    错误: " << e.what() << std::endl;
            r.build_time = -1;
        }
        return r;
    }
    
    double computeRecall(const std::vector<faiss::idx_t>& pred_labels, 
                        const std::vector<faiss::idx_t>& true_labels, 
                        int top_k) {
        double recall_sum = 0.0;
        
        for (int i = 0; i < nq; i++) {
            std::unordered_set<faiss::idx_t> true_set;
            true_set.reserve(top_k * 2);
            for (int j = 0; j < top_k; j++) {
                true_set.insert(true_labels[i * k + j]);
            }
            int hit = 0;
            for (int j = 0; j < top_k; j++) {
                if (true_set.find(pred_labels[i * k + j]) != true_set.end()) ++hit;
            }
            if (top_k > 0) {
                recall_sum += static_cast<double>(hit) / top_k;
            }
        }
        
        return recall_sum / nq;
    }
    
    std::vector<TestResult> runIndexTypeBenchmarks(const Options& opt) {
        std::vector<TestResult> results;
        auto contains = [&](const std::string& what) {
            if (opt.which.size() == 1 && opt.which[0] == "all") return true;
            return std::find(opt.which.begin(), opt.which.end(), what) != opt.which.end();
        };

        if (contains("hnsw_flat")) {
            auto r = testHNSWFlat(opt.hnsw_M, opt.efC, opt.efS);
            if (r.build_time >= 0) results.push_back(r);
        }
        if (contains("hnsw_sq")) {
            auto r = testHNSWSQ(opt.sq_qtype, opt.hnsw_M, opt.efC, opt.efS);
            if (r.build_time >= 0) results.push_back(r);
        }
        if (contains("hnsw_pq")) {
            auto r = testHNSWPQ(opt.pq_m, opt.pq_nbits, opt.hnsw_M, opt.efC, opt.efS, opt.pq_train_ratio);
            if (r.build_time >= 0) results.push_back(r);
        }
        return results;
    }

    void printAnalysis(const std::vector<TestResult>& results) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📈 HNSW Index Types 指标总结" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        if (results.empty()) {
            std::cout << "❌ 没有成功的测试结果" << std::endl;
            return;
        }
        // Best by recall and fastest
        auto best_recall = *std::max_element(results.begin(), results.end(), [](const TestResult& a, const TestResult& b){ return a.recall_10 < b.recall_10; });
        auto fastest = *std::min_element(results.begin(), results.end(), [](const TestResult& a, const TestResult& b){ return a.search_time_ms < b.search_time_ms; });
    std::cout << "🎯 最高召回: " << best_recall.method << "(" << best_recall.params << ") R@" << k << "=" << std::fixed << std::setprecision(3) << best_recall.recall_10 << std::endl;
        std::cout << "⚡ 最快搜索: " << fastest.method << "(" << fastest.params << ") " << std::setprecision(2) << fastest.search_time_ms << " ms/query" << std::endl;

        // Detailed table
    std::cout << "\n📊 详细结果:" << std::endl;
    std::string rcol = std::string("R@") + std::to_string(k);
    std::cout << std::setw(12) << "Method"
                  << std::setw(30) << "Params"
                  << std::setw(8) << "efS"
                  << std::setw(10) << "Train(s)"
                  << std::setw(10) << "Add(s)"
                  << std::setw(10) << "Build(s)"
                  << std::setw(12) << "Search(ms)"
          << std::setw(10) << rcol
                  << std::setw(12) << "Mem(MB)"
                  << std::setw(10) << "Compress" << std::endl;
        std::cout << std::string(130, '-') << std::endl;
        for (const auto& r : results) {
            std::cout << std::setw(12) << r.method
                      << std::setw(30) << r.params
                      << std::setw(8) << r.efS
                      << std::setw(10) << std::fixed << std::setprecision(2) << r.train_time
                      << std::setw(10) << std::setprecision(2) << r.add_time
                      << std::setw(10) << std::setprecision(2) << r.build_time
                      << std::setw(12) << std::setprecision(2) << r.search_time_ms
                      << std::setw(10) << std::setprecision(3) << r.recall_10
                      << std::setw(12) << std::setprecision(1) << r.memory_mb
                      << std::setw(10) << std::setprecision(1) << r.compression_ratio
                      << std::endl;
        }
    }

    void saveResults(const std::vector<TestResult>& results, const std::string& path) {
        std::ofstream csv_file(path);
        if (csv_file.is_open()) {
            csv_file << "method,params,hnsw_M,efC,efS,pq_m,pq_nbits,qtype,train_time,add_time,build_time,search_time_ms,recall_at_1,recall_at_5,recall_at_10,memory_mb,compression_ratio\n";
            for (const auto& r : results) {
                csv_file << r.method << "," << r.params << ","
                         << r.hnsw_M << "," << r.efC << "," << r.efS << ","
                         << r.pq_m << "," << r.pq_nbits << "," << qtypeName(r.qtype) << ","
                         << r.train_time << "," << r.add_time << "," << r.build_time << "," << r.search_time_ms << ","
                         << r.recall_1 << "," << r.recall_5 << "," << r.recall_10 << ","
                         << r.memory_mb << "," << r.compression_ratio << "\n";
            }
            csv_file.close();
            std::cout << "\n💾 结果已保存到: " << path << std::endl;
        } else {
            std::cerr << "❌ 无法创建CSV文件" << std::endl;
        }
    }
};

static int parseQType(const std::string& s) {
    using QT = faiss::ScalarQuantizer::QuantizerType;
    if (s == "QT_8bit") return (int)QT::QT_8bit;
    if (s == "QT_4bit") return (int)QT::QT_4bit;
    if (s == "QT_8bit_uniform") return (int)QT::QT_8bit_uniform;
    if (s == "QT_fp16") return (int)QT::QT_fp16;
    // fallback: try int
    try { return std::stoi(s); } catch (...) { return (int)QT::QT_8bit; }
}

static std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, ',')) { if (!item.empty()) out.push_back(item); }
    return out;
}

int main(int argc, char** argv) {
    Options opt;
    // Naive CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&](int& dst){ if (i + 1 < argc) dst = std::stoi(argv[++i]); };
        auto nexts = [&](std::string& dst){ if (i + 1 < argc) dst = argv[++i]; };
        if (a == "--dim") next(opt.dim);
        else if (a == "--nb") next(opt.nb);
        else if (a == "--nq") next(opt.nq);
        else if (a == "--k") next(opt.k);
        else if (a == "--hnsw-M") next(opt.hnsw_M);
        else if (a == "--efC") next(opt.efC);
        else if (a == "--efS") next(opt.efS);
        else if (a == "--pq-m") next(opt.pq_m);
        else if (a == "--pq-nbits") next(opt.pq_nbits);
        else if (a == "--pq-train-ratio") {
            std::string s; nexts(s);
            try { opt.pq_train_ratio = std::stod(s); } catch (...) {}
            if (opt.pq_train_ratio < 0.0) opt.pq_train_ratio = 0.0;
            if (opt.pq_train_ratio > 1.0) opt.pq_train_ratio = 1.0;
        }
        else if (a == "--sq-qtype") { std::string s; nexts(s); opt.sq_qtype = parseQType(s); }
        else if (a == "--which") { std::string s; nexts(s); opt.which = splitCSV(s); }
        else if (a == "--out-csv") nexts(opt.out_csv);
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: ./bench [options]\n"
                         "  --dim INT           vector dim (default 128)\n"
                         "  --nb INT            database size (default 20000)\n"
                         "  --nq INT            queries count (default 500)\n"
                         "  --k INT             top-k (default 10)\n"
                         "  --hnsw-M INT        HNSW M (default 32)\n"
                         "  --efC INT           HNSW efConstruction (default 200)\n"
                         "  --efS INT           HNSW efSearch (default 64)\n"
                         "  --pq-m INT          PQ m for HNSW-PQ (default 16)\n"
                         "  --pq-nbits INT      PQ nbits for HNSW-PQ (default 8)\n"
                         "  --pq-train-ratio F  PQ train ratio in [0,1] (default 1.0)\n"
                         "  --sq-qtype STR      SQ qtype for HNSW-SQ (QT_8bit|QT_4bit|QT_8bit_uniform|QT_fp16)\n"
                         "  --which STR         comma list: all|hnsw_flat|hnsw_sq|hnsw_pq (default all)\n"
                         "  --out-csv PATH      output csv path (default hnsw_index_types_results.csv)\n";
            return 0;
        }
    }

    try {
        PQBenchmark bench(opt.dim, opt.nb, opt.nq, opt.k);
        auto results = bench.runIndexTypeBenchmarks(opt);
        bench.printAnalysis(results);
        bench.saveResults(results, opt.out_csv);
        std::cout << "\n✅ 分析完成！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl; return 1;
    }
    return 0;
}
