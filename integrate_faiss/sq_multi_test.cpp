#include "faiss/IndexHNSW.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/MetricType.h"
#include "faiss/impl/ScalarQuantizer.h"
#include <faiss/IndexFlat.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <set>
#include <numeric>
#include <cmath>

// 统计结果结构体
struct TestResult {
    std::vector<double> build_times;
    std::vector<double> train_times; 
    std::vector<double> search_times;
    std::vector<double> recalls;
    
    void add_result(double build_time, double train_time, double search_time, double recall) {
        build_times.push_back(build_time);
        train_times.push_back(train_time);
        search_times.push_back(search_time);
        recalls.push_back(recall);
    }
    
    double avg_build_time() const { return std::accumulate(build_times.begin(), build_times.end(), 0.0) / build_times.size(); }
    double avg_train_time() const { return std::accumulate(train_times.begin(), train_times.end(), 0.0) / train_times.size(); }
    double avg_search_time() const { return std::accumulate(search_times.begin(), search_times.end(), 0.0) / search_times.size(); }
    double avg_recall() const { return std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size(); }
    
    double std_build_time() const { return calculate_std(build_times, avg_build_time()); }
    double std_train_time() const { return calculate_std(train_times, avg_train_time()); }
    double std_search_time() const { return calculate_std(search_times, avg_search_time()); }
    double std_recall() const { return calculate_std(recalls, avg_recall()); }
    
private:
    double calculate_std(const std::vector<double>& values, double mean) const {
        if (values.size() <= 1) return 0.0;
        double sum = 0.0;
        for (double value : values) {
            sum += (value - mean) * (value - mean);
        }
        return std::sqrt(sum / (values.size() - 1));
    }
};

// 计算召回率
double calculate_recall(const std::vector<faiss::idx_t>& I_baseline, 
                       const std::vector<faiss::idx_t>& I_test, 
                       size_t nq, int k) {
    int total_hits = 0;
    for (size_t qi = 0; qi < nq; ++qi) {
        std::set<faiss::idx_t> baseline_set;
        std::set<faiss::idx_t> test_set;
        
        for (int j = 0; j < k; ++j) {
            baseline_set.insert(I_baseline[qi * k + j]);
            test_set.insert(I_test[qi * k + j]);
        }
        
        for (auto id : test_set) {
            if (baseline_set.find(id) != baseline_set.end()) {
                total_hits++;
            }
        }
    }
    return (double)total_hits / (nq * k);
}

void print_stats(const std::string& name, const TestResult& result) {
    std::cout << name << ":" << std::endl;
    std::cout << "  训练时间: " << std::fixed << std::setprecision(1) 
              << result.avg_train_time() << " ± " << result.std_train_time() << " ms" << std::endl;
    std::cout << "  构建时间: " << std::fixed << std::setprecision(1) 
              << result.avg_build_time() << " ± " << result.std_build_time() << " ms" << std::endl;
    std::cout << "  总时间: " << std::fixed << std::setprecision(1) 
              << (result.avg_train_time() + result.avg_build_time()) << " ms" << std::endl;
    std::cout << "  搜索时间: " << std::fixed << std::setprecision(1) 
              << result.avg_search_time() << " ± " << result.std_search_time() << " μs" << std::endl;
    std::cout << "  召回率: " << std::fixed << std::setprecision(2) 
              << result.avg_recall() * 100 << " ± " << result.std_recall() * 100 << "%" << std::endl;
}

int main() {
    // 测试参数
    const int num_runs = 5;  // 运行次数，可根据需要调整
    
    // 向量维度
    int d = 128;
    // MaxDegree
    int M = 32;
    // 数据量
    size_t nb = 10000;
    size_t nq = 100;

    std::cout << "=== 多次运行测试 (运行 " << num_runs << " 次) ===" << std::endl;
    std::cout << "数据规模: " << nb << " 向量, 维度: " << d << std::endl;
    
    // 原始数据内存使用
    size_t original_memory = nb * d * sizeof(float);
    std::cout << "原始数据内存: " << std::fixed << std::setprecision(2) 
              << original_memory / (1024.0 * 1024.0) << " MB" << std::endl;

    // 存储测试结果
    TestResult flat_result, hnsw_result, hnsw_sq8_result, hnsw_sq4_result;

    // 预先生成基准查询结果
    std::vector<faiss::idx_t> I_flat_baseline;
    
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "\n--- 运行 " << (run + 1) << "/" << num_runs << " ---" << std::endl;
        
        // 构造随机数据（每次运行使用不同的seed保证独立性）
        std::mt19937 rng(42 + run);
        std::uniform_real_distribution<float> dist(0.f, 1.f);

        std::vector<float> xb(nb * d);
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = dist(rng);

        // 查询向量
        std::vector<float> xq(nq * d);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = dist(rng);

        // 1. IndexFlat (基准)
        faiss::IndexFlatL2 index_flat(d);
        
        auto start = std::chrono::high_resolution_clock::now();
        index_flat.add(nb, xb.data());
        auto end = std::chrono::high_resolution_clock::now();
        auto build_time_flat = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // 搜索测试
        int k = 10;
        std::vector<faiss::idx_t> I_flat(nq * k);
        std::vector<float> D_flat(nq * k);
        
        start = std::chrono::high_resolution_clock::now();
        index_flat.search(nq, xq.data(), k, D_flat.data(), I_flat.data());
        end = std::chrono::high_resolution_clock::now();
        auto search_time_flat = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (run == 0) {
            I_flat_baseline = I_flat;  // 第一次运行作为基准
        }
        
        flat_result.add_result(build_time_flat, 0, search_time_flat, 1.0);  // 基准召回率为100%
        
        // 2. HNSW (无量化)
        faiss::IndexHNSWFlat index_hnsw(d, M);
        index_hnsw.hnsw.efConstruction = 200;
        index_hnsw.hnsw.efSearch = 50;
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw.add(nb, xb.data());
        end = std::chrono::high_resolution_clock::now();
        auto build_time_hnsw = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::vector<faiss::idx_t> I_hnsw(nq * k);
        std::vector<float> D_hnsw(nq * k);
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw.search(nq, xq.data(), k, D_hnsw.data(), I_hnsw.data());
        end = std::chrono::high_resolution_clock::now();
        auto search_time_hnsw = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double recall_hnsw = calculate_recall(I_flat_baseline, I_hnsw, nq, k);
        hnsw_result.add_result(build_time_hnsw, 0, search_time_hnsw, recall_hnsw);
        
        // 3. HNSW + SQ8
        faiss::IndexHNSWSQ index_hnsw_sq8(d, faiss::ScalarQuantizer::QT_8bit, M);
        index_hnsw_sq8.hnsw.efConstruction = 200;
        index_hnsw_sq8.hnsw.efSearch = 50;
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq8.train(nb, xb.data());
        end = std::chrono::high_resolution_clock::now();
        auto train_time_hnsw_sq8 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq8.add(nb, xb.data());
        end = std::chrono::high_resolution_clock::now();
        auto build_time_hnsw_sq8 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::vector<faiss::idx_t> I_hnsw_sq8(nq * k);
        std::vector<float> D_hnsw_sq8(nq * k);
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq8.search(nq, xq.data(), k, D_hnsw_sq8.data(), I_hnsw_sq8.data());
        end = std::chrono::high_resolution_clock::now();
        auto search_time_hnsw_sq8 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double recall_hnsw_sq8 = calculate_recall(I_flat_baseline, I_hnsw_sq8, nq, k);
        hnsw_sq8_result.add_result(build_time_hnsw_sq8, train_time_hnsw_sq8, search_time_hnsw_sq8, recall_hnsw_sq8);
        
        // 4. HNSW + SQ4
        faiss::IndexHNSWSQ index_hnsw_sq4(d, faiss::ScalarQuantizer::QT_4bit, M);
        index_hnsw_sq4.hnsw.efConstruction = 200;
        index_hnsw_sq4.hnsw.efSearch = 50;
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq4.train(nb, xb.data());
        end = std::chrono::high_resolution_clock::now();
        auto train_time_hnsw_sq4 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq4.add(nb, xb.data());
        end = std::chrono::high_resolution_clock::now();
        auto build_time_hnsw_sq4 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::vector<faiss::idx_t> I_hnsw_sq4(nq * k);
        std::vector<float> D_hnsw_sq4(nq * k);
        
        start = std::chrono::high_resolution_clock::now();
        index_hnsw_sq4.search(nq, xq.data(), k, D_hnsw_sq4.data(), I_hnsw_sq4.data());
        end = std::chrono::high_resolution_clock::now();
        auto search_time_hnsw_sq4 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double recall_hnsw_sq4 = calculate_recall(I_flat_baseline, I_hnsw_sq4, nq, k);
        hnsw_sq4_result.add_result(build_time_hnsw_sq4, train_time_hnsw_sq4, search_time_hnsw_sq4, recall_hnsw_sq4);
        
        std::cout << "完成运行 " << (run + 1) << std::endl;
    }

    // 打印综合结果
    std::cout << "\n=== 综合统计结果 (基于 " << num_runs << " 次运行) ===" << std::endl;
    
    print_stats("IndexFlat (基准)", flat_result);
    print_stats("HNSW (无量化)", hnsw_result);
    print_stats("HNSW + SQ8", hnsw_sq8_result);
    print_stats("HNSW + SQ4", hnsw_sq4_result);

    // 内存分析
    size_t sq8_memory = nb * d * sizeof(uint8_t);
    size_t sq4_memory = nb * d / 2;
    
    std::cout << "\n=== 内存分析 ===" << std::endl;
    std::cout << "原始向量内存: " << std::fixed << std::setprecision(2) 
              << original_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "SQ8 量化内存: " << std::fixed << std::setprecision(2) 
              << sq8_memory / (1024.0 * 1024.0) << " MB (节省 " 
              << std::setprecision(1) << (1.0 - (double)sq8_memory / original_memory) * 100 << "%)" << std::endl;
    std::cout << "SQ4 量化内存: " << std::fixed << std::setprecision(2) 
              << sq4_memory / (1024.0 * 1024.0) << " MB (节省 " 
              << std::setprecision(1) << (1.0 - (double)sq4_memory / original_memory) * 100 << "%)" << std::endl;

    // 性能对比分析
    std::cout << "\n=== 性能对比分析 ===" << std::endl;
    std::cout << "相对于 IndexFlat 的性能提升:" << std::endl;
    std::cout << "HNSW + SQ8:" << std::endl;
    std::cout << "  搜索速度: " << std::fixed << std::setprecision(1) 
              << flat_result.avg_search_time() / hnsw_sq8_result.avg_search_time() << "x" << std::endl;
    std::cout << "  召回率: " << std::fixed << std::setprecision(1) 
              << hnsw_sq8_result.avg_recall() * 100 << "%" << std::endl;
    std::cout << "  总时间比: " << std::fixed << std::setprecision(1) 
              << (hnsw_sq8_result.avg_train_time() + hnsw_sq8_result.avg_build_time()) / flat_result.avg_build_time() << "x" << std::endl;
    
    std::cout << "HNSW + SQ4:" << std::endl;
    std::cout << "  搜索速度: " << std::fixed << std::setprecision(1) 
              << flat_result.avg_search_time() / hnsw_sq4_result.avg_search_time() << "x" << std::endl;
    std::cout << "  召回率: " << std::fixed << std::setprecision(1) 
              << hnsw_sq4_result.avg_recall() * 100 << "%" << std::endl;
    std::cout << "  总时间比: " << std::fixed << std::setprecision(1) 
              << (hnsw_sq4_result.avg_train_time() + hnsw_sq4_result.avg_build_time()) / flat_result.avg_build_time() << "x" << std::endl;

    return 0;
}
