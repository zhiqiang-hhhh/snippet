#include <faiss/IndexPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <set>

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
        std::cout << "=== PQ量化参数M影响分析 C++ Demo ===" << std::endl;
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
        int M;
        int sub_dim;
        int clusters_per_subvector;
        double build_time;
        double search_time_ms;
        double recall_1;
        double recall_5;
        double recall_10;
        double memory_mb;
        double compression_ratio;
    };
    
    TestResult testPQIndex(int m, int nbits) {
        std::cout << "  测试 IndexPQ (M=" << m << ", nbits=" << nbits << ")..." << std::endl;
        
        TestResult result;
        result.M = m;
        result.sub_dim = dim / m;
        result.clusters_per_subvector = 1 << nbits;  // 2^nbits
        
        try {
            // 创建PQ索引
            faiss::IndexPQ index(dim, m, nbits, faiss::METRIC_L2);
            
            // 训练和添加向量
            auto start = std::chrono::high_resolution_clock::now();
            index.train(nb, database.data());
            index.add(nb, database.data());
            auto end = std::chrono::high_resolution_clock::now();
            
            result.build_time = std::chrono::duration<double>(end - start).count();
            
            // 搜索测试
            std::vector<float> distances(nq * k);
            std::vector<faiss::idx_t> labels(nq * k);
            
            start = std::chrono::high_resolution_clock::now();
            index.search(nq, queries.data(), k, distances.data(), labels.data());
            end = std::chrono::high_resolution_clock::now();
            
            result.search_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / nq;
            
            // 计算召回率
            result.recall_1 = computeRecall(labels, ground_truth, 1);
            result.recall_5 = computeRecall(labels, ground_truth, 5);
            result.recall_10 = computeRecall(labels, ground_truth, k);
            
            // 内存使用估算
            result.memory_mb = static_cast<double>(nb * m * nbits) / 8.0 / (1024 * 1024);
            result.compression_ratio = static_cast<double>(nb * dim * 4) / (nb * m * nbits / 8.0);
            
        } catch (const std::exception& e) {
            std::cerr << "    错误: " << e.what() << std::endl;
            // 设置默认值表示失败
            result.build_time = -1;
            result.search_time_ms = -1;
            result.recall_1 = result.recall_5 = result.recall_10 = 0;
            result.memory_mb = 0;
            result.compression_ratio = 0;
        }
        
        return result;
    }
    
    TestResult testHNSWPQIndex(int m, int nbits, int hnsw_m = 16) {
        std::cout << "  测试 IndexHNSWPQ (M=" << m << ", nbits=" << nbits << ")..." << std::endl;
        
        TestResult result;
        result.M = m;
        result.sub_dim = dim / m;
        result.clusters_per_subvector = 1 << nbits;
        
        try {
            // 创建HNSW+PQ索引
            faiss::IndexHNSWPQ index(dim, m, nbits, hnsw_m, faiss::METRIC_L2);
            index.hnsw.efConstruction = 100;  // 降低构建参数
            
            // 训练和添加向量
            auto start = std::chrono::high_resolution_clock::now();
            index.train(nb, database.data());
            index.add(nb, database.data());
            auto end = std::chrono::high_resolution_clock::now();
            
            result.build_time = std::chrono::duration<double>(end - start).count();
            
            // 搜索测试
            index.hnsw.efSearch = 32;
            std::vector<float> distances(nq * k);
            std::vector<faiss::idx_t> labels(nq * k);
            
            start = std::chrono::high_resolution_clock::now();
            index.search(nq, queries.data(), k, distances.data(), labels.data());
            end = std::chrono::high_resolution_clock::now();
            
            result.search_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / nq;
            
            // 计算召回率
            result.recall_1 = computeRecall(labels, ground_truth, 1);
            result.recall_5 = computeRecall(labels, ground_truth, 5);
            result.recall_10 = computeRecall(labels, ground_truth, k);
            
            // 内存使用估算 (PQ码 + HNSW图)
            double pq_memory = static_cast<double>(nb * m * nbits) / 8.0;
            double hnsw_memory = static_cast<double>(nb * hnsw_m * 8);  // 近似
            result.memory_mb = (pq_memory + hnsw_memory) / (1024 * 1024);
            result.compression_ratio = static_cast<double>(nb * dim * 4) / (pq_memory + hnsw_memory);
            
        } catch (const std::exception& e) {
            std::cerr << "    错误: " << e.what() << std::endl;
            result.build_time = -1;
            result.search_time_ms = -1;
            result.recall_1 = result.recall_5 = result.recall_10 = 0;
            result.memory_mb = 0;
            result.compression_ratio = 0;
        }
        
        return result;
    }
    
    double computeRecall(const std::vector<faiss::idx_t>& pred_labels, 
                        const std::vector<faiss::idx_t>& true_labels, 
                        int top_k) {
        double recall_sum = 0.0;
        
        for (int i = 0; i < nq; i++) {
            std::set<faiss::idx_t> true_set, pred_set;
            
            for (int j = 0; j < top_k; j++) {
                true_set.insert(true_labels[i * k + j]);
                pred_set.insert(pred_labels[i * k + j]);
            }
            
            // 计算交集
            std::vector<faiss::idx_t> intersection;
            std::set_intersection(true_set.begin(), true_set.end(),
                                pred_set.begin(), pred_set.end(),
                                std::back_inserter(intersection));
            
            if (true_set.size() > 0) {
                recall_sum += static_cast<double>(intersection.size()) / true_set.size();
            }
        }
        
        return recall_sum / nq;
    }
    
    void runBenchmark() {
        std::cout << "\n开始PQ参数M影响分析..." << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // 选择合适的参数范围
        std::vector<int> m_values = {8, 16, 32, 64};  // M值需要能整除dim
        int nbits = 4;  // 使用4位，每个子量化器16个聚类中心
        
        std::cout << "测试参数:" << std::endl;
        std::cout << "  M值范围: ";
        for (int m : m_values) {
            std::cout << m << " ";
        }
        std::cout << std::endl;
        std::cout << "  nbits: " << nbits << " (每个子量化器 " << (1 << nbits) << " 个聚类中心)" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::vector<TestResult> pq_results, hnswpq_results;
        
        for (int m : m_values) {
            if (dim % m != 0) {
                std::cout << "跳过 M=" << m << ": 维度" << dim << "不能被M整除" << std::endl;
                continue;
            }
            
            std::cout << "\n🔬 测试 M=" << m << " (子向量维度: " << dim/m << ")" << std::endl;
            
            // 检查聚类中心数量是否合理
            int total_clusters = (1 << nbits);  // 每个子量化器的聚类中心数
            if (nb < total_clusters * 10) {  // 至少需要10倍的训练数据
                std::cout << "  ⚠️  警告: 训练数据可能不足 (需要 " << total_clusters << " 个聚类中心)" << std::endl;
            }
            
            // 测试PQ索引
            TestResult pq_result = testPQIndex(m, nbits);
            if (pq_result.build_time >= 0) {  // 成功
                pq_results.push_back(pq_result);
                std::cout << "    PQ: 构建=" << std::fixed << std::setprecision(2) << pq_result.build_time 
                         << "s, 搜索=" << pq_result.search_time_ms << "ms, 召回率@10=" 
                         << std::setprecision(3) << pq_result.recall_10 << std::endl;
            }
            
            // 测试HNSW+PQ索引
            TestResult hnswpq_result = testHNSWPQIndex(m, nbits);
            if (hnswpq_result.build_time >= 0) {  // 成功
                hnswpq_results.push_back(hnswpq_result);
                std::cout << "    HNSW+PQ: 构建=" << std::fixed << std::setprecision(2) << hnswpq_result.build_time 
                         << "s, 搜索=" << hnswpq_result.search_time_ms << "ms, 召回率@10=" 
                         << std::setprecision(3) << hnswpq_result.recall_10 << std::endl;
            }
        }
        
        // 输出结果分析
        printAnalysis(pq_results, hnswpq_results);
        
        // 保存结果到CSV
        saveResults(pq_results, hnswpq_results);
    }
    
    void printAnalysis(const std::vector<TestResult>& pq_results, 
                      const std::vector<TestResult>& hnswpq_results) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📈 PQ量化参数M影响分析总结" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        if (pq_results.empty() && hnswpq_results.empty()) {
            std::cout << "❌ 没有成功的测试结果" << std::endl;
            return;
        }
        
        std::cout << "\n📊 主要发现:" << std::endl;
        
        // PQ结果分析
        if (!pq_results.empty()) {
            auto best_pq_recall = *std::max_element(pq_results.begin(), pq_results.end(),
                [](const TestResult& a, const TestResult& b) { return a.recall_10 < b.recall_10; });
            auto fastest_pq = *std::min_element(pq_results.begin(), pq_results.end(),
                [](const TestResult& a, const TestResult& b) { return a.search_time_ms < b.search_time_ms; });
            
            std::cout << "🎯 PQ最佳召回率: M=" << best_pq_recall.M 
                     << ", 召回率=" << std::fixed << std::setprecision(3) << best_pq_recall.recall_10 << std::endl;
            std::cout << "⚡ PQ最快搜索: M=" << fastest_pq.M 
                     << ", 时间=" << std::setprecision(2) << fastest_pq.search_time_ms << "ms" << std::endl;
        }
        
        // HNSW+PQ结果分析
        if (!hnswpq_results.empty()) {
            auto best_hnsw_recall = *std::max_element(hnswpq_results.begin(), hnswpq_results.end(),
                [](const TestResult& a, const TestResult& b) { return a.recall_10 < b.recall_10; });
            auto fastest_hnsw = *std::min_element(hnswpq_results.begin(), hnswpq_results.end(),
                [](const TestResult& a, const TestResult& b) { return a.search_time_ms < b.search_time_ms; });
            
            std::cout << "🎯 HNSW+PQ最佳召回率: M=" << best_hnsw_recall.M 
                     << ", 召回率=" << std::setprecision(3) << best_hnsw_recall.recall_10 << std::endl;
            std::cout << "⚡ HNSW+PQ最快搜索: M=" << fastest_hnsw.M 
                     << ", 时间=" << std::setprecision(2) << fastest_hnsw.search_time_ms << "ms" << std::endl;
        }
        
        std::cout << "\n📋 关键趋势:" << std::endl;
        std::cout << "• M增加 → 子向量维度减少 → 量化精度可能降低" << std::endl;
        std::cout << "• M增加 → 子量化器数量增加 → 训练时间增加" << std::endl;
        std::cout << "• M适中时通常有最佳的精度/速度平衡" << std::endl;
        
        // 详细结果表格
        std::cout << "\n📊 详细结果 (PQ):" << std::endl;
        std::cout << std::setw(5) << "M" << std::setw(10) << "子维度" << std::setw(12) << "构建时间(s)" 
                 << std::setw(12) << "搜索时间(ms)" << std::setw(12) << "召回率@10" << std::setw(12) << "内存(MB)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (const auto& result : pq_results) {
            std::cout << std::setw(5) << result.M 
                     << std::setw(10) << result.sub_dim
                     << std::setw(12) << std::fixed << std::setprecision(2) << result.build_time
                     << std::setw(12) << std::setprecision(2) << result.search_time_ms
                     << std::setw(12) << std::setprecision(3) << result.recall_10
                     << std::setw(12) << std::setprecision(1) << result.memory_mb << std::endl;
        }
        
        if (!hnswpq_results.empty()) {
            std::cout << "\n📊 详细结果 (HNSW+PQ):" << std::endl;
            std::cout << std::setw(5) << "M" << std::setw(10) << "子维度" << std::setw(12) << "构建时间(s)" 
                     << std::setw(12) << "搜索时间(ms)" << std::setw(12) << "召回率@10" << std::setw(12) << "内存(MB)" << std::endl;
            std::cout << std::string(65, '-') << std::endl;
            
            for (const auto& result : hnswpq_results) {
                std::cout << std::setw(5) << result.M 
                         << std::setw(10) << result.sub_dim
                         << std::setw(12) << std::fixed << std::setprecision(2) << result.build_time
                         << std::setw(12) << std::setprecision(2) << result.search_time_ms
                         << std::setw(12) << std::setprecision(3) << result.recall_10
                         << std::setw(12) << std::setprecision(1) << result.memory_mb << std::endl;
            }
        }
    }
    
    void saveResults(const std::vector<TestResult>& pq_results, 
                    const std::vector<TestResult>& hnswpq_results) {
        std::ofstream csv_file("pq_benchmark_results.csv");
        
        if (csv_file.is_open()) {
            // CSV头部
            csv_file << "Index_Type,M,Sub_Dim,Clusters_Per_Subvector,Build_Time,Search_Time_MS,"
                    << "Recall_1,Recall_5,Recall_10,Memory_MB,Compression_Ratio\n";
            
            // PQ结果
            for (const auto& result : pq_results) {
                csv_file << "PQ," << result.M << "," << result.sub_dim << "," 
                        << result.clusters_per_subvector << "," << result.build_time << ","
                        << result.search_time_ms << "," << result.recall_1 << ","
                        << result.recall_5 << "," << result.recall_10 << ","
                        << result.memory_mb << "," << result.compression_ratio << "\n";
            }
            
            // HNSW+PQ结果
            for (const auto& result : hnswpq_results) {
                csv_file << "HNSW+PQ," << result.M << "," << result.sub_dim << "," 
                        << result.clusters_per_subvector << "," << result.build_time << ","
                        << result.search_time_ms << "," << result.recall_1 << ","
                        << result.recall_5 << "," << result.recall_10 << ","
                        << result.memory_mb << "," << result.compression_ratio << "\n";
            }
            
            csv_file.close();
            std::cout << "\n💾 结果已保存到: pq_benchmark_results.csv" << std::endl;
        } else {
            std::cerr << "❌ 无法创建CSV文件" << std::endl;
        }
    }
};

int main() {
    try {
        // 创建基准测试实例
        PQBenchmark benchmark(
            128,    // dim - 向量维度
            20000,  // nb - 数据库向量数量
            500,    // nq - 查询向量数量
            10      // k - 搜索邻居数
        );
        
        // 运行基准测试
        benchmark.runBenchmark();
        
        std::cout << "\n✅ 分析完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
