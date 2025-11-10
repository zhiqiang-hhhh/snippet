#include "faiss/IndexPreTransform.h"
#include "faiss/VectorTransform.h"
#include "faiss/index_factory.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#ifdef HAVE_VECML
#include "vecml_shim/vecml_shim.h"
#endif

// 通过序列化来估算索引内存占用：写到临时文件，取文件大小（MB）
static double measureIndexSerializedSize(faiss::Index *index) {
  if (!index)
    return 0.0;
  char tmpl[] = "/tmp/faiss_index_memXXXXXX";
  int fd = mkstemp(tmpl);
  if (fd == -1) {
    return 0.0;
  }
  close(fd); // 只保留文件名
  try {
    faiss::write_index(index, tmpl);
    struct stat st;
    double mb = 0.0;
    if (::stat(tmpl, &st) == 0) {
      mb = (double)st.st_size / (1024.0 * 1024.0);
    }
    ::unlink(tmpl);
    return mb;
  } catch (...) {
    ::unlink(tmpl);
    return 0.0;
  }
}

struct Options {
  int dim = 128;
  int nb = 20000;
  int nq = 500;
  int k = 10;

  // Support list variants (if non-empty, they override the single value)
  std::vector<int> dim_list;
  std::vector<int> nb_list;
  std::vector<int> nq_list;
  std::vector<int> k_list;

  // HNSW
  int hnsw_M = 32;
  int efC = 200;
  int efS = 64;
  std::vector<int> hnsw_M_list;
  std::vector<int> efC_list;
  std::vector<int> efS_list;

  // PQ (for HNSW-PQ)
  int pq_m = 16;
  int pq_nbits = 8;
  double pq_train_ratio = 1.0; // fraction of nb used for PQ training [0,1]
  std::vector<int> pq_m_list;
  std::vector<int> pq_nbits_list;
  std::vector<double> pq_train_ratio_list; // allow multiple training ratios

  // SQ (for HNSW-SQ) - removed, now specified in which parameter
  // int sq_qtype = (int)faiss::ScalarQuantizer::QuantizerType::QT_8bit;
  // std::vector<int> sq_qtype_list;

  // Which tests to run: all, hnsw_flat, hnsw_sq4, hnsw_sq8, hnsw_pq, hnsw_rabitq,
  // ivf_flat, ivf_sq4, ivf_sq8, ivf_pq, ivf_rbq, rabitq
  std::vector<std::string> which = {"all"};

  // IVF parameters (for IVF-Flat / IVF-PQ / IVF-SQ)
  int ivf_nlist = 256;
  int ivf_nprobe = 8;
  std::vector<int> ivf_nlist_list;
  std::vector<int> ivf_nprobe_list;

  // RaBitQ (for HNSW/IVF-RaBitQ)
  int rabitq_qb = 8;            // query quantization bits
  bool rabitq_centered = false; // whether to center queries
  std::vector<int> rabitq_qb_list;
  std::vector<int> rabitq_centered_list;

  // RaBitQ Refine (for IVF-RaBitQ refine two-stage)
  // refine type: flat|sq8|sq4|fp16|bf16 (bf16 falls back to fp16 if
  // unsupported)
  std::string rbq_refine_type = "flat";
  std::vector<std::string> rbq_refine_type_list;
  int rbq_refine_k = 2;
  std::vector<int> rbq_refine_k_list;

  // Output
  std::string out_csv = "hnsw_index_types_results.csv";
  std::string save_data_dir = ""; // if not empty, save test data to this directory
  std::string load_data_dir = ""; // if not empty, load test data from this directory

  // VecML options
  std::string vecml_base_path = ""; // path to vecml SDK dir (contains include/lib)
  std::string vecml_license_path = "license.txt";

  // Whether to build transposed centroids for PQ (optimization)
  bool transpose_centroid = false;
};

class PQBenchmark {
private:
  int dim;
  int nb;
  int nq;
  int k;
  bool
      transpose_centroid; // control whether to sync transposed centroids for PQ
  std::vector<float> database;
  std::vector<float> queries;
  std::vector<faiss::idx_t> ground_truth;
  std::string save_data_dir; // directory to save test data

public:
  PQBenchmark(int dim = 128, int nb = 20000, int nq = 500, int k = 10,
              bool transpose_centroid = false, const std::string& data_dir = "",
              const std::string& load_dir = "")
      : dim(dim), nb(nb), nq(nq), k(k), transpose_centroid(transpose_centroid), save_data_dir(data_dir) {
    std::cout << "=== HNSW Index Types Benchmark (C++) ===" << std::endl;
    std::cout << "配置: dim=" << dim << ", nb=" << nb << ", nq=" << nq
              << ", k=" << k << std::endl;
    
    if (!load_dir.empty()) {
      std::cout << "从 " << load_dir << " 加载测试数据..." << std::endl;
      if (loadTestData(load_dir)) {
        std::cout << "数据加载成功!" << std::endl;
      } else {
        std::cout << "数据加载失败，生成新数据..." << std::endl;
        generateData();
        computeGroundTruth();
      }
    } else {
      generateData();
      computeGroundTruth();
    }
    
    if (!save_data_dir.empty()) {
      saveTestData();
    }
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

  void normalizeVectors(float *vectors, int n, int d) {
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

  void saveTestData() {
    if (save_data_dir.empty()) return;
    
    // Create directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + save_data_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
      std::cerr << "警告: 无法创建目录 " << save_data_dir << std::endl;
      return;
    }
    
    std::cout << "保存测试数据到 " << save_data_dir << "..." << std::endl;
    
    // Save database vectors
    std::string db_file = save_data_dir + "/database_" + std::to_string(dim) + "d_" + std::to_string(nb) + "n.fvecs";
    saveFVecs(db_file, database, nb, dim);
    
    // Save query vectors  
    std::string query_file = save_data_dir + "/queries_" + std::to_string(dim) + "d_" + std::to_string(nq) + "n.fvecs";
    saveFVecs(query_file, queries, nq, dim);
    
    // Save ground truth
    std::string gt_file = save_data_dir + "/groundtruth_" + std::to_string(nq) + "q_" + std::to_string(k) + "k.ivecs";
    saveIVecs(gt_file, ground_truth, nq, k);
    
    std::cout << "数据已保存: " << db_file << ", " << query_file << ", " << gt_file << std::endl;
  }

  void saveFVecs(const std::string& filename, const std::vector<float>& data, int n, int d) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "错误: 无法创建文件 " << filename << std::endl;
      return;
    }
    
    for (int i = 0; i < n; i++) {
      file.write(reinterpret_cast<const char*>(&d), sizeof(int));
      file.write(reinterpret_cast<const char*>(data.data() + i * d), d * sizeof(float));
    }
    file.close();
  }

  void saveIVecs(const std::string& filename, const std::vector<faiss::idx_t>& data, int n, int d) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "错误: 无法创建文件 " << filename << std::endl;
      return;
    }
    
    for (int i = 0; i < n; i++) {
      file.write(reinterpret_cast<const char*>(&d), sizeof(int));
      for (int j = 0; j < d; j++) {
        int val = static_cast<int>(data[i * d + j]);
        file.write(reinterpret_cast<const char*>(&val), sizeof(int));
      }
    }
    file.close();
  }

  bool loadTestData(const std::string& load_dir) {
    try {
      // 构建文件路径
      std::string db_file = load_dir + "/database_" + std::to_string(dim) + "d_" + std::to_string(nb) + "n.fvecs";
      std::string query_file = load_dir + "/queries_" + std::to_string(dim) + "d_" + std::to_string(nq) + "n.fvecs";
      std::string gt_file = load_dir + "/groundtruth_" + std::to_string(nq) + "q_" + std::to_string(k) + "k.ivecs";
      
      // 检查文件是否存在
      if (!fileExists(db_file) || !fileExists(query_file) || !fileExists(gt_file)) {
        std::cout << "部分文件不存在，需要的文件:" << std::endl;
        std::cout << "  " << db_file << std::endl;
        std::cout << "  " << query_file << std::endl; 
        std::cout << "  " << gt_file << std::endl;
        return false;
      }
      
      // 加载数据
      if (!loadFVecs(db_file, database, nb, dim)) {
        std::cerr << "加载数据库向量失败: " << db_file << std::endl;
        return false;
      }
      
      if (!loadFVecs(query_file, queries, nq, dim)) {
        std::cerr << "加载查询向量失败: " << query_file << std::endl;
        return false;
      }
      
      if (!loadIVecs(gt_file, ground_truth, nq, k)) {
        std::cerr << "加载ground truth失败: " << gt_file << std::endl;
        return false;
      }
      
      std::cout << "成功加载: " << nb << " 个数据库向量, " << nq << " 个查询向量, ground truth" << std::endl;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "加载数据时发生异常: " << e.what() << std::endl;
      return false;
    }
  }

  bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
  }

  bool loadFVecs(const std::string& filename, std::vector<float>& data, int expected_n, int expected_d) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }
    
    data.resize(expected_n * expected_d);
    
    for (int i = 0; i < expected_n; i++) {
      int d;
      file.read(reinterpret_cast<char*>(&d), sizeof(int));
      if (file.fail() || d != expected_d) {
        std::cerr << "维度不匹配: 期望 " << expected_d << ", 实际 " << d << " (在向量 " << i << ")" << std::endl;
        return false;
      }
      
      file.read(reinterpret_cast<char*>(data.data() + i * expected_d), expected_d * sizeof(float));
      if (file.fail()) {
        std::cerr << "读取向量数据失败: 向量 " << i << std::endl;
        return false;
      }
    }
    
    file.close();
    return true;
  }

  bool loadIVecs(const std::string& filename, std::vector<faiss::idx_t>& data, int expected_n, int expected_d) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }
    
    data.resize(expected_n * expected_d);
    
    for (int i = 0; i < expected_n; i++) {
      int d;
      file.read(reinterpret_cast<char*>(&d), sizeof(int));
      if (file.fail() || d != expected_d) {
        std::cerr << "k值不匹配: 期望 " << expected_d << ", 实际 " << d << " (在查询 " << i << ")" << std::endl;
        return false;
      }
      
      for (int j = 0; j < expected_d; j++) {
        int val;
        file.read(reinterpret_cast<char*>(&val), sizeof(int));
        if (file.fail()) {
          std::cerr << "读取ground truth失败: 查询 " << i << ", 位置 " << j << std::endl;
          return false;
        }
        data[i * expected_d + j] = static_cast<faiss::idx_t>(val);
      }
    }
    
    file.close();
    return true;
  }

  struct TestResult {
    std::string method; // HNSW-Flat / HNSW-SQ / HNSW-PQ
    // Dataset level
    int dim = 0;
    int nb = 0;
    int nq = 0;
    int k = 0;
    int hnsw_M = 0;
    int efC = 0;
    int efS = 0;
    int ivf_nlist = 0; // IVF params
    int ivf_nprobe = 0;
    int pq_m = 0;
    int pq_nbits = 0;
    double pq_train_ratio = 1.0; // valid for PQ rows
    int qtype = -1;              // for SQ
    int rabitq_qb = 0;           // for RaBitQ
    int rabitq_centered = 0;     // 0/1 for printing
    std::string refine_type;     // rbq_refine only
    int refine_k = 0;            // rbq_refine only

    double train_time = 0.0;     // seconds
    double add_time = 0.0;       // seconds
    double build_time = 0.0;     // train + add seconds
    double search_time_ms = 0.0; // per query
    double recall_1 = 0.0;
    double recall_5 = 0.0;
    double recall_10 = 0.0;
    double mbs_on_disk = 0.0; // serialized index size in MB
    double compression_ratio = 0.0;
  };

  // 小工具：qtype 名称
  static const char *qtypeName(int qtype) {
    using QT = faiss::ScalarQuantizer::QuantizerType;
    switch (qtype) {
    case QT::QT_8bit:
      return "QT_8bit";
    case QT::QT_4bit:
      return "QT_4bit";
    case QT::QT_8bit_uniform:
      return "QT_8bit_uniform";
    case QT::QT_fp16:
      return "QT_fp16";
#ifdef FAISS_HAVE_QT_8BIT_DIRECT
    case QT::QT_8bit_direct:
      return "QT_8bit_direct";
#endif
    default:
      return "unknown";
    }
  }

  // 小工具：获取qtype对应的bit数
  static int getQTypeBits(int qtype) {
    using QT = faiss::ScalarQuantizer::QuantizerType;
    switch (qtype) {
    case QT::QT_8bit:
    case QT::QT_8bit_uniform:
#ifdef FAISS_HAVE_QT_8BIT_DIRECT
    case QT::QT_8bit_direct:
#endif
      return 8;
    case QT::QT_4bit:
      return 4;
    case QT::QT_fp16:
      return 16;
    default:
      return 8;
    }
  }

  // HNSW-Flat
  TestResult testHNSWFlat(int hnsw_m, int efC, int efS) {
    std::cout << "  测试 IndexHNSWFlat (M=" << hnsw_m << ", efC=" << efC
              << ", efS=" << efS << ")..." << std::endl;
    TestResult r{};
    r.method = "HNSW-Flat";
    r.hnsw_M = hnsw_m;
    r.efC = efC;
    r.efS = efS;
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
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.hnsw.efSearch = efS;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      auto t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      auto t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      // Serialize index to a temporary file to measure real memory footprint
      // (including graph + codes)
      r.mbs_on_disk = measureIndexSerializedSize(&index);
      // For flat, compression ratio = original (dim*4) /
      // (serialized_size_per_vector)
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 1.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // HNSW-SQ
  TestResult testHNSWSQ(int qtype, int hnsw_m, int efC, int efS) {
    std::string qtype_name = qtypeName(qtype);
    std::cout << "  测试 IndexHNSWSQ (" << qtype_name << ", M=" << hnsw_m
              << ", efC=" << efC << ", efS=" << efS << ")..." << std::endl;
    TestResult r{};
    r.method = "HNSW-SQ" + std::to_string(getQTypeBits(qtype));
    r.hnsw_M = hnsw_m;
    r.efC = efC;
    r.efS = efS;
    r.qtype = qtype;
    try {
      faiss::IndexHNSWSQ index(dim,
                               (faiss::ScalarQuantizer::QuantizerType)qtype,
                               hnsw_m, faiss::METRIC_L2);
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
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.hnsw.efSearch = efS;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      int bits = 8;
      using QT = faiss::ScalarQuantizer::QuantizerType;
      switch ((QT)qtype) {
      case QT::QT_8bit:
      case QT::QT_8bit_uniform:
        bits = 8;
        break;
      case QT::QT_4bit:
        bits = 4;
        break;
      case QT::QT_fp16:
        bits = 16;
        break;
      default:
        bits = 8;
        break;
      }
      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // HNSW-PQ
  TestResult testHNSWPQ(int pq_m, int pq_nbits, int hnsw_m, int efC, int efS,
                        double train_ratio) {
    std::cout << "  测试 IndexHNSWPQ (M=" << hnsw_m << ", PQ(m=" << pq_m
              << ", nbits=" << pq_nbits << "), efC=" << efC << ", efS=" << efS
              << ", tr=" << train_ratio << ")..." << std::endl;
    TestResult r{};
    r.method = "HNSW-PQ";
    r.hnsw_M = hnsw_m;
    r.efC = efC;
    r.efS = efS;
    r.pq_m = pq_m;
    r.pq_nbits = pq_nbits;
    r.pq_train_ratio = train_ratio;
    try {
      if (dim % pq_m != 0) {
        std::cout << "    跳过：dim=" << dim << " 不能被 m=" << pq_m << " 整除"
                  << std::endl;
        r.build_time = -1;
        return r;
      }
      faiss::IndexHNSWPQ index(dim, pq_m, hnsw_m, pq_nbits);
      index.hnsw.efConstruction = efC;
      // Decide training subset size based on ratio
      int train_n = (int)std::llround((double)nb * train_ratio);
      if (train_n < 1)
        train_n = 1;
      if (train_n > nb)
        train_n = nb;
      if (train_n < nb) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << (double)train_n / (double)nb;
        std::cout << "    使用训练样本: " << train_n << "/" << nb
                  << " (ratio=" << oss.str() << ")" << std::endl;
      }
      auto t0 = std::chrono::high_resolution_clock::now();
      index.train(train_n, database.data());
      if (transpose_centroid) {
        faiss::IndexPQ *index_pq = static_cast<faiss::IndexPQ *>(index.storage);
        if (index_pq) {
          index_pq->pq.sync_transposed_centroids();
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      r.train_time = std::chrono::duration<double>(t1 - t0).count();
      auto t2 = std::chrono::high_resolution_clock::now();
      index.add(nb, database.data());
      auto t3 = std::chrono::high_resolution_clock::now();
      r.add_time = std::chrono::duration<double>(t3 - t2).count();
      r.build_time = r.train_time + r.add_time;
      std::cout << std::fixed << std::setprecision(3)
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.hnsw.efSearch = efS;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // HNSW-RaBitQ
  TestResult testHNSWRaBitQ(int hnsw_m, int efC, int efS, int qb,
                            bool centered) {
    std::cout << "  测试 IndexHNSWRaBitQ (M=" << hnsw_m << ", efC=" << efC
              << ", efS=" << efS << ", qb=" << qb
              << (centered ? ", centered" : "") << ")..." << std::endl;
    TestResult r{};
    r.method = "HNSW-RaBitQ";
    r.hnsw_M = hnsw_m;
    r.efC = efC;
    r.efS = efS;
    r.rabitq_qb = qb;
    r.rabitq_centered = centered ? 1 : 0;
    try {
      faiss::IndexHNSWRaBitQ index(dim, hnsw_m, faiss::METRIC_L2);
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

      index.set_query_quantization(qb, centered);

      faiss::SearchParametersHNSWRaBitQ sp;
      sp.efSearch = efS;
      sp.qb = (uint8_t)qb;
      sp.centered = centered;

      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data(), &sp);
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
      std::cout << std::fixed << std::setprecision(3)
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // IVF-Flat
  TestResult testIVFFlat(int nlist, int nprobe) {
    std::cout << "  测试 IndexIVFFlat (nlist=" << nlist << ", nprobe=" << nprobe
              << ")..." << std::endl;
    TestResult r{};
    r.method = "IVF-Flat";
    r.ivf_nlist = nlist;
    r.ivf_nprobe = nprobe;
    try {
      faiss::IndexFlatL2 coarse(dim);
      faiss::IndexIVFFlat index(&coarse, dim, nlist, faiss::METRIC_L2);
      // need train
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
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.nprobe = nprobe;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      // For IVF-Flat, vectors stored in float -> no compression
      r.compression_ratio = 1.0; // mark as 1.0; will be printed as NA below
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // IVF-PQ
  TestResult testIVFPQ(int nlist, int nprobe, int m, int nbits,
                       double train_ratio) {
    std::cout << "  测试 IndexIVFPQ (nlist=" << nlist << ", nprobe=" << nprobe
              << ", PQ(m=" << m << ", nbits=" << nbits << ")"
              << ", tr=" << train_ratio << ")..." << std::endl;
    TestResult r{};
    r.method = "IVF-PQ";
    r.ivf_nlist = nlist;
    r.ivf_nprobe = nprobe;
    r.pq_m = m;
    r.pq_nbits = nbits;
    r.pq_train_ratio = train_ratio;
    try {
      if (dim % m != 0) {
        std::cout << "    跳过：dim=" << dim << " 不能被 m=" << m << " 整除"
                  << std::endl;
        r.build_time = -1;
        return r;
      }
      faiss::IndexFlatL2 coarse(dim);
      faiss::IndexIVFPQ index(&coarse, dim, nlist, m, nbits, faiss::METRIC_L2);
      // train subset
      int train_n = (int)std::llround((double)nb * train_ratio);
      if (train_n < 1)
        train_n = 1;
      if (train_n > nb)
        train_n = nb;
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
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.nprobe = nprobe;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // IVF-SQ
  TestResult testIVFSQ(int nlist, int nprobe, int qtype) {
    std::string qtype_name = qtypeName(qtype);
    std::cout << "  测试 IndexIVFScalarQuantizer (" << qtype_name
              << ", nlist=" << nlist << ", nprobe=" << nprobe << ")..."
              << std::endl;
    TestResult r{};
    r.method = "IVF-SQ" + std::to_string(getQTypeBits(qtype));
    r.ivf_nlist = nlist;
    r.ivf_nprobe = nprobe;
    r.qtype = qtype;
    try {
      faiss::IndexFlatL2 coarse(dim);
      faiss::IndexIVFScalarQuantizer index(
          &coarse, dim, nlist, (faiss::ScalarQuantizer::QuantizerType)qtype,
          faiss::METRIC_L2);
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
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      index.nprobe = nprobe;
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data());
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // IVF-RaBitQ
  TestResult testIVFRaBitQ(int nlist, int nprobe, int qb, bool centered) {
    std::cout << "  测试 IndexIVFRaBitQ (nlist=" << nlist
              << ", nprobe=" << nprobe << ", qb=" << qb
              << ", centered=" << (centered ? "true" : "false") << ")..."
              << std::endl;
    TestResult r{};
    r.method = "IVF-RaBitQ";
    r.ivf_nlist = nlist;
    r.ivf_nprobe = nprobe;
    r.rabitq_qb = qb;
    r.rabitq_centered = centered ? 1 : 0;
    try {
      std::unique_ptr<faiss::IndexFlat> coarse =
          std::make_unique<faiss::IndexFlat>(dim, faiss::METRIC_L2);
      std::unique_ptr<faiss::IndexIVFRaBitQ> index =
          std::make_unique<faiss::IndexIVFRaBitQ>(coarse.release(), dim, nlist,
                                                  faiss::METRIC_L2);
      index->own_fields = true;
      index->qb = (uint8_t)qb; // default query bits

      // Try both with and without rotation for better recall
      auto rr = std::make_unique<faiss::RandomRotationMatrix>(dim, dim);
      // Align with bench_rabitq.py: explicit init of random rotation
      rr->init(123);
      auto idx_rr = std::make_unique<faiss::IndexPreTransform>(rr.release(),
                                                               index.release());
      idx_rr->own_fields = true;

      auto t0 = std::chrono::high_resolution_clock::now();
      // Use more training data if available for better quantization
      int train_size = std::min(nb, std::max(50000, nb / 2));
      idx_rr->train(train_size, database.data());
      auto t1 = std::chrono::high_resolution_clock::now();
      r.train_time = std::chrono::duration<double>(t1 - t0).count();
      auto t2 = std::chrono::high_resolution_clock::now();
      idx_rr->add(nb, database.data());
      auto t3 = std::chrono::high_resolution_clock::now();
      r.add_time = std::chrono::duration<double>(t3 - t2).count();
      r.build_time = r.train_time + r.add_time;
      std::cout << std::fixed << std::setprecision(3)
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time 
                << " s, 训练样本=" << train_size << std::endl;

      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      faiss::IVFRaBitQSearchParameters sp;
      sp.qb = (uint8_t)qb;
      sp.centered = centered;
      // Important: when passing SearchParameters, nprobe from params is used.
      // Default is 1, so make sure to pass the desired nprobe here.
      sp.nprobe = nprobe;
      t0 = std::chrono::high_resolution_clock::now();
      idx_rr->search(nq, queries.data(), k, distances.data(), labels.data(),
                     &sp);
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      std::cout << "    调试: nprobe=" << sp.nprobe << ", qb=" << (int)sp.qb 
                << ", centered=" << sp.centered << std::endl;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(idx_rr.get());
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // IVF-RaBitQ + Refine (two-stage)
  TestResult testIVFRaBitQRefine(int nlist, int nprobe, int qb, bool centered,
                                 const std::string &refine_type, int refine_k) {
    std::cout << "  测试 IVFRaBitQ+Refine (nlist=" << nlist
              << ", nprobe=" << nprobe << ", qb=" << qb
              << ", centered=" << (centered ? "true" : "false")
              << ", refine_type=" << refine_type << ", refine_k=" << refine_k
              << ")..." << std::endl;
    TestResult r{};
    r.method = "IVF-RaBitQ-Refine";
    r.ivf_nlist = nlist;
    r.ivf_nprobe = nprobe;
    r.rabitq_qb = qb;
    r.rabitq_centered = centered ? 1 : 0;
    r.refine_type = refine_type;
    r.refine_k = refine_k;
    try {
      // Base: RR -> IVFRaBitQ
      std::unique_ptr<faiss::IndexFlatL2> coarse =
          std::make_unique<faiss::IndexFlatL2>(dim);
      std::unique_ptr<faiss::IndexIVFRaBitQ> base_ivf =
          std::make_unique<faiss::IndexIVFRaBitQ>(coarse.release(), dim, nlist,
                                                  faiss::METRIC_L2);
      base_ivf->nprobe = nprobe;
      base_ivf->own_fields = true;
      base_ivf->qb = (uint8_t)qb; // default query bits

      auto rr = std::make_unique<faiss::RandomRotationMatrix>(dim, dim);
      rr->init(123);
      std::unique_ptr<faiss::IndexPreTransform> base =
          std::make_unique<faiss::IndexPreTransform>(rr.release(),
                                                     base_ivf.release());
      base->own_fields = true;

      // Refine index by type (operates on original vectors)
      std::unique_ptr<faiss::Index> refine;
      auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
      };
      std::string rt = to_lower(refine_type);
      if (rt == "flat") {
        refine = std::make_unique<faiss::IndexFlatL2>(dim);
      } else if (rt == "sq8") {
        refine = std::make_unique<faiss::IndexScalarQuantizer>(
            dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            faiss::METRIC_L2);
      } else if (rt == "sq4") {
        refine = std::make_unique<faiss::IndexScalarQuantizer>(
            dim, faiss::ScalarQuantizer::QuantizerType::QT_4bit,
            faiss::METRIC_L2);
      } else if (rt == "fp16" || rt == "bf16") {
        // Use fp16 SQ as a proxy; if bf16 unsupported in this FAISS build, fall
        // back to fp16
        refine = std::make_unique<faiss::IndexScalarQuantizer>(
            dim, faiss::ScalarQuantizer::QuantizerType::QT_fp16,
            faiss::METRIC_L2);
      } else {
        std::cout << "    未知 refine_type='" << refine_type << "'，回退到 flat"
                  << std::endl;
        refine = std::make_unique<faiss::IndexFlatL2>(dim);
      }

      // Wrap with IndexRefine
      std::unique_ptr<faiss::IndexRefine> wrapper =
          std::make_unique<faiss::IndexRefine>(base.release(),
                                               refine.release());
      wrapper->own_fields = true;

      // Train both indices as needed
      auto t0 = std::chrono::high_resolution_clock::now();
      wrapper->train(nb, database.data());
      auto t1 = std::chrono::high_resolution_clock::now();
      r.train_time = std::chrono::duration<double>(t1 - t0).count();
      auto t2 = std::chrono::high_resolution_clock::now();
      wrapper->add(nb, database.data());
      auto t3 = std::chrono::high_resolution_clock::now();
      r.add_time = std::chrono::duration<double>(t3 - t2).count();
      r.build_time = r.train_time + r.add_time;
      std::cout << std::fixed << std::setprecision(3)
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;

      // Search with two-stage params
      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      faiss::IVFRaBitQSearchParameters ivf_sp;
      ivf_sp.qb = (uint8_t)qb;
      ivf_sp.centered = centered;
      ivf_sp.nprobe = nprobe;
      // nprobe is already set on index; still pass via base params
      faiss::IndexRefineSearchParameters ref_sp;
      ref_sp.k_factor = refine_k;
      ref_sp.base_index_params = &ivf_sp;
      t0 = std::chrono::high_resolution_clock::now();
      wrapper->search(nq, queries.data(), k, distances.data(), labels.data(),
                      &ref_sp);
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(wrapper.get());
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

  // RaBitQ (flat)
  TestResult testRaBitQ(int qb, bool centered) {
    std::cout << "  测试 IndexRaBitQ (qb=" << qb
              << ", centered=" << (centered ? "true" : "false") << ")..."
              << std::endl;
    TestResult r{};
    r.method = "RaBitQ";
    r.rabitq_qb = qb;
    r.rabitq_centered = centered ? 1 : 0;
    try {
      faiss::IndexRaBitQ index(dim, faiss::METRIC_L2);
      // train
      auto t0 = std::chrono::high_resolution_clock::now();
      index.train(nb, database.data());
      auto t1 = std::chrono::high_resolution_clock::now();
      r.train_time = std::chrono::duration<double>(t1 - t0).count();
      // add
      auto t2 = std::chrono::high_resolution_clock::now();
      index.add(nb, database.data());
      auto t3 = std::chrono::high_resolution_clock::now();
      r.add_time = std::chrono::duration<double>(t3 - t2).count();
      r.build_time = r.train_time + r.add_time;

      std::vector<float> distances(nq * k);
      std::vector<faiss::idx_t> labels(nq * k);
      faiss::RaBitQSearchParameters sp;
      sp.qb = (uint8_t)qb;
      sp.centered = centered;
      t0 = std::chrono::high_resolution_clock::now();
      index.search(nq, queries.data(), k, distances.data(), labels.data(), &sp);
      t1 = std::chrono::high_resolution_clock::now();
      r.search_time_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / nq;

      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      r.mbs_on_disk = measureIndexSerializedSize(&index);
      if (nb > 0) {
        double per_vec = (r.mbs_on_disk * 1024.0 * 1024.0) / nb;
        r.compression_ratio = ((double)dim * 4.0) / per_vec;
      } else {
        r.compression_ratio = 0.0;
      }
      std::cout << std::fixed << std::setprecision(3)
                << "    训练(train)= " << r.train_time
                << " s, 建索引(add)= " << r.add_time
                << " s, 总构建= " << r.build_time << " s" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

#ifdef HAVE_VECML
  // VecML test using the C shim (vecml_shim)
  TestResult testVecML(const std::string &base_path, const std::string &license_path) {
    TestResult r{};
    r.method = "VecML";
    r.dim = dim;
    r.nb = nb;
    r.nq = nq;
    r.k = k;
    // create ctx
    vecml_ctx_t ctx = vecml_create(base_path.c_str(), license_path.c_str());
    if (!ctx) {
      std::cerr << "VecML: failed to create context" << std::endl;
      r.build_time = -1;
      return r;
    }
    // VecML does not have a separate train phase. Measure add_time and set
    // build_time = add_time to reflect that "build" is just adding data.
    try {
      auto t_add0 = std::chrono::high_resolution_clock::now();
      int add_ret = vecml_add_data(ctx, database.data(), nb, dim, nullptr);
      auto t_add1 = std::chrono::high_resolution_clock::now();
      if (add_ret != 0) {
        std::cerr << "VecML: add_data failed: " << add_ret << std::endl;
        vecml_destroy(ctx);
        r.build_time = -1;
        return r;
      }
      r.add_time = std::chrono::duration<double>(t_add1 - t_add0).count();
      r.train_time = 0.0; // no train step in VecML
      r.build_time = r.add_time;

      // Search timing
      std::vector<long> out_ids((size_t)nq * k, -1);
      auto t_s0 = std::chrono::high_resolution_clock::now();
      int sret = vecml_search(ctx, queries.data(), nq, dim, k, out_ids.data());
      auto t_s1 = std::chrono::high_resolution_clock::now();
      if (sret != 0) {
        std::cerr << "VecML: search failed: " << sret << std::endl;
        vecml_destroy(ctx);
        r.build_time = -1;
        return r;
      }
      double search_sec = std::chrono::duration<double>(t_s1 - t_s0).count();
      if (nq > 0)
        r.search_time_ms = (search_sec / (double)nq) * 1000.0;
      else
        r.search_time_ms = 0.0;

      // convert to faiss labels
      std::vector<faiss::idx_t> labels((size_t)nq * k);
      for (int i = 0; i < nq * k; ++i) {
        labels[i] = out_ids[i] >= 0 ? static_cast<faiss::idx_t>(out_ids[i]) : -1;
      }
      r.recall_1 = computeRecall(labels, ground_truth, 1);
      r.recall_5 = computeRecall(labels, ground_truth, 5);
      r.recall_10 = computeRecall(labels, ground_truth, k);

      vecml_destroy(ctx);
      return r;
    } catch (const std::exception &e) {
      std::cerr << "VecML: exception during add/search: " << e.what() << std::endl;
      vecml_destroy(ctx);
      r.build_time = -1;
      return r;
    }
    
  }
#endif

  double computeRecall(const std::vector<faiss::idx_t> &pred_labels,
                       const std::vector<faiss::idx_t> &true_labels,
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
        if (true_set.find(pred_labels[i * k + j]) != true_set.end())
          ++hit;
      }
      if (top_k > 0) {
        recall_sum += static_cast<double>(hit) / top_k;
      }
    }

    return recall_sum / nq;
  }

  std::vector<TestResult> runIndexTypeBenchmarks(const Options &opt) {
    std::vector<TestResult> results;
    auto contains = [&](const std::string &what) {
      if (opt.which.size() == 1 && opt.which[0] == "all")
        return true;
      return std::find(opt.which.begin(), opt.which.end(), what) !=
             opt.which.end();
    };

    // Helper to check for SQ variants
    auto containsSQ = [&](const std::string &base, int bits) {
      std::string target = base + std::to_string(bits);
      return contains(target);
    };

    // Parse SQ types from which parameter
    auto getSQTypes = [&](const std::string &base) {
      std::vector<int> types;
      if (contains("all") || containsSQ(base, 4)) {
        types.push_back((int)faiss::ScalarQuantizer::QuantizerType::QT_4bit);
      }
      if (contains("all") || containsSQ(base, 8)) {
        types.push_back((int)faiss::ScalarQuantizer::QuantizerType::QT_8bit);
      }
      return types;
    };

    // Prepare lists (fallback to single value if list empty)
    auto prepareIntList = [](const std::vector<int> &lst, int single) {
      return lst.empty() ? std::vector<int>{single} : lst;
    };
    auto prepareDoubleList = [](const std::vector<double> &lst, double single) {
      return lst.empty() ? std::vector<double>{single} : lst;
    };

    auto dim_list = prepareIntList(opt.dim_list, opt.dim);
    auto nb_list = prepareIntList(opt.nb_list, opt.nb);
    auto nq_list = prepareIntList(opt.nq_list, opt.nq);
    auto k_list = prepareIntList(opt.k_list, opt.k);
    auto hnsw_M_list = prepareIntList(opt.hnsw_M_list, opt.hnsw_M);
    auto efC_list = prepareIntList(opt.efC_list, opt.efC);
    auto efS_list = prepareIntList(opt.efS_list, opt.efS);
    auto pq_m_list = prepareIntList(opt.pq_m_list, opt.pq_m);
    auto pq_nbits_list = prepareIntList(opt.pq_nbits_list, opt.pq_nbits);
    auto pq_train_ratio_list =
        prepareDoubleList(opt.pq_train_ratio_list, opt.pq_train_ratio);
    // auto sq_qtype_list = prepareIntList(opt.sq_qtype_list, opt.sq_qtype); // removed
    auto ivf_nlist_list = prepareIntList(opt.ivf_nlist_list, opt.ivf_nlist);
    auto ivf_nprobe_list = prepareIntList(opt.ivf_nprobe_list, opt.ivf_nprobe);
    auto rabitq_qb_list = prepareIntList(opt.rabitq_qb_list, opt.rabitq_qb);
    auto rabitq_centered_list =
        prepareIntList(opt.rabitq_centered_list, opt.rabitq_centered ? 1 : 0);
    auto rbq_refine_type_list =
        (opt.rbq_refine_type_list.empty()
             ? std::vector<std::string>{opt.rbq_refine_type}
             : opt.rbq_refine_type_list);
    auto rbq_refine_k_list =
        prepareIntList(opt.rbq_refine_k_list, opt.rbq_refine_k);

    // Iterate over dataset-level combinations first: dim, nb, nq, k
    for (int dim_v : dim_list) {
      for (int nb_v : nb_list) {
        for (int nq_v : nq_list) {
          for (int k_v : k_list) {
            // Rebuild benchmark data for each dataset combo
            PQBenchmark bench_local(dim_v, nb_v, nq_v, k_v,
                                    opt.transpose_centroid, opt.save_data_dir, opt.load_data_dir);
            // HNSW param combos
            for (int hM : hnsw_M_list) {
              for (int efC_v : efC_list) {
                for (int efS_v : efS_list) {
                  if (contains("hnsw_flat")) {
                    auto r = bench_local.testHNSWFlat(hM, efC_v, efS_v);
                    if (r.build_time >= 0) {
                      r.dim = dim_v;
                      r.nb = nb_v;
                      r.nq = nq_v;
                      r.k = k_v;
                      results.push_back(r);
                    }
                  }
                  // HNSW-SQ with specific bit types
                  auto hnsw_sq_types = getSQTypes("hnsw_sq");
                  for (int qtype_v : hnsw_sq_types) {
                    auto r = bench_local.testHNSWSQ(qtype_v, hM, efC_v, efS_v);
                    if (r.build_time >= 0) {
                      r.dim = dim_v;
                      r.nb = nb_v;
                      r.nq = nq_v;
                      r.k = k_v;
                      results.push_back(r);
                    }
                  }
                  if (contains("hnsw_pq")) {
                    for (int m_v : pq_m_list) {
                      for (int nbits_v : pq_nbits_list) {
                        for (double tr_v : pq_train_ratio_list) {
                          auto r = bench_local.testHNSWPQ(m_v, nbits_v, hM,
                                                          efC_v, efS_v, tr_v);
                          if (r.build_time >= 0) {
                            r.dim = dim_v;
                            r.nb = nb_v;
                            r.nq = nq_v;
                            r.k = k_v;
                            r.pq_train_ratio = tr_v;
                            results.push_back(r);
                          }
                        }
                      }
                    }
                  }
                  if (contains("hnsw_rabitq")) {
                    for (int qb_v : rabitq_qb_list) {
                      for (int centered_flag : rabitq_centered_list) {
                        bool centered = centered_flag != 0;
                        auto r = bench_local.testHNSWRaBitQ(hM, efC_v, efS_v,
                                                            qb_v, centered);
                        if (r.build_time >= 0) {
                          r.dim = dim_v;
                          r.nb = nb_v;
                          r.nq = nq_v;
                          r.k = k_v;
                          results.push_back(r);
                        }
                      }
                    }
                  }
                }
              }
            }
            // IVF param combos
            for (int nl_v : ivf_nlist_list) {
              for (int np_v : ivf_nprobe_list) {
                if (contains("ivf_flat")) {
                  auto r = bench_local.testIVFFlat(nl_v, np_v);
                  if (r.build_time >= 0) {
                    r.dim = dim_v;
                    r.nb = nb_v;
                    r.nq = nq_v;
                    r.k = k_v;
                    results.push_back(r);
                  }
                }
                // IVF-SQ with specific bit types
                auto ivf_sq_types = getSQTypes("ivf_sq");
                for (int qtype_v : ivf_sq_types) {
                  auto r = bench_local.testIVFSQ(nl_v, np_v, qtype_v);
                  if (r.build_time >= 0) {
                    r.dim = dim_v;
                    r.nb = nb_v;
                    r.nq = nq_v;
                    r.k = k_v;
                    results.push_back(r);
                  }
                }
                if (contains("ivf_pq")) {
                  for (int m_v : pq_m_list) {
                    for (int nbits_v : pq_nbits_list) {
                      for (double tr_v : pq_train_ratio_list) {
                        auto r = bench_local.testIVFPQ(nl_v, np_v, m_v, nbits_v,
                                                       tr_v);
                        if (r.build_time >= 0) {
                          r.dim = dim_v;
                          r.nb = nb_v;
                          r.nq = nq_v;
                          r.k = k_v;
                          r.pq_train_ratio = tr_v;
                          results.push_back(r);
                        }
                      }
                    }
                  }
                }
                if (contains("ivf_rbq")) {
                  for (int qb_v : rabitq_qb_list) {
                    for (int centered_flag : rabitq_centered_list) {
                      bool centered = centered_flag != 0;
                      auto r =
                          bench_local.testIVFRaBitQ(nl_v, np_v, qb_v, centered);
                      if (r.build_time >= 0) {
                        r.dim = dim_v;
                        r.nb = nb_v;
                        r.nq = nq_v;
                        r.k = k_v;
                        results.push_back(r);
                      }
                    }
                  }
                }
                // IVFRaBitQ + Refine
                if (contains("rbq_refine") || contains("ivf_rbq_refine")) {
                  for (int qb_v : rabitq_qb_list) {
                    for (int centered_flag : rabitq_centered_list) {
                      bool centered = centered_flag != 0;
                      for (const auto &rt : rbq_refine_type_list) {
                        for (int rk : rbq_refine_k_list) {
                          if (rk < 1)
                            continue;
                          auto r = bench_local.testIVFRaBitQRefine(
                              nl_v, np_v, qb_v, centered, rt, rk);
                          if (r.build_time >= 0) {
                            r.dim = dim_v;
                            r.nb = nb_v;
                            r.nq = nq_v;
                            r.k = k_v;
                            results.push_back(r);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            // Direct RaBitQ (flat) combos (independent of IVF/HNSW params)
            if (contains("rabitq")) {
              for (int qb_v : rabitq_qb_list) {
                for (int centered_flag : rabitq_centered_list) {
                  bool centered = centered_flag != 0;
                  auto r = bench_local.testRaBitQ(qb_v, centered);
                  if (r.build_time >= 0) {
                    r.dim = dim_v;
                    r.nb = nb_v;
                    r.nq = nq_v;
                    r.k = k_v;
                    results.push_back(r);
                  }
                }
              }
            }
          }
        }
      }
    }
    return results;
  }

  void printAnalysis(const std::vector<TestResult> &results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "📈 HNSW Index Types 指标总结" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    if (results.empty()) {
      std::cout << "❌ 没有成功的测试结果" << std::endl;
      return;
    }
    // Best by recall and fastest
    auto best_recall =
        *std::max_element(results.begin(), results.end(),
                          [](const TestResult &a, const TestResult &b) {
                            return a.recall_10 < b.recall_10;
                          });
    auto fastest =
        *std::min_element(results.begin(), results.end(),
                          [](const TestResult &a, const TestResult &b) {
                            return a.search_time_ms < b.search_time_ms;
                          });
    auto fmtParams = [](const TestResult &r) {
      std::ostringstream oss;
      oss << "M=" << r.hnsw_M;
      if (r.method == "HNSW-SQ") {
        oss << "," << PQBenchmark::qtypeName(r.qtype);
      } else if (r.method == "HNSW-PQ") {
        oss << ",PQ(m=" << r.pq_m << ",nbits=" << r.pq_nbits << ")";
      } else if (r.method == "HNSW-RaBitQ") {
        oss << ",qb=" << r.rabitq_qb;
        if (r.rabitq_centered) {
          oss << ",centered";
        }
      }
      oss << ",efC=" << r.efC << ",efS=" << r.efS;
      return oss.str();
    };
    std::cout << "🎯 最高召回: " << best_recall.method << "("
              << fmtParams(best_recall) << ") R@" << best_recall.k << "="
              << std::fixed << std::setprecision(3) << best_recall.recall_10
              << std::endl;
    std::cout << "⚡ 最快搜索: " << fastest.method << "(" << fmtParams(fastest)
              << ") " << std::setprecision(2) << fastest.search_time_ms
              << " ms/query" << std::endl;

    // Detailed table
    std::cout << "\n📊 详细结果:" << std::endl;
    // Print header matching CSV columns and order (space after each column)
    std::cout << std::left
              << std::setw(18) << "method" << ' '
              << std::setw(6) << "dim" << ' '
              << std::setw(8) << "nb" << ' '
              << std::setw(6) << "nq" << ' '
              << std::setw(4) << "k" << ' '
              << std::setw(8) << "hnsw_M" << ' '
              << std::setw(6) << "efC" << ' '
              << std::setw(6) << "efS" << ' '
              << std::setw(10) << "ivf_nlist" << ' '
              << std::setw(10) << "ivf_nprobe" << ' '
              << std::setw(6) << "pq_m" << ' '
              << std::setw(10) << "pq_nbits" << ' '
              << std::setw(10) << "rbq_qb" << ' '
              << std::setw(12) << "rbq_centered" << ' '
              << std::setw(10) << "refine_k" << ' '
              << std::setw(12) << "refine_type" << ' '
              << std::setw(10) << "qtype" << ' '
              << std::setw(12) << "train_tm" << ' '
              << std::setw(10) << "add_tm" << ' '
              << std::setw(12) << "build_tm" << ' '
              << std::setw(14) << "search_tm_ms" << ' '
              << std::setw(10) << "r@1" << ' '
              << std::setw(10) << "r@5" << ' '
              << std::setw(10) << "r@k" << ' '
              << std::setw(14) << "mbs_on_disk" << ' '
              << std::setw(13) << "compression" << std::endl;
    // Separator line length: sum of widths + spaces between columns (26 columns total)
    std::cout << std::string(
                     (18 + 6 + 8 + 6 + 4 + 8 + 6 + 6 + 10 + 10 + 6 + 10 + 10 +
                      12 + 10 + 12 + 10 + 12 + 10 + 12 + 14 + 10 + 10 + 10 +
                      14 + 13) + 25,
                     '-')
              << std::endl;    for (const auto &r : results) {
      std::cout << std::left
                << std::setw(18) << r.method << ' '
                << std::setw(6) << r.dim << ' '
                << std::setw(8) << r.nb << ' '
                << std::setw(6) << r.nq << ' '
                << std::setw(4) << r.k << ' '
                << std::setw(8) << r.hnsw_M << ' '
                << std::setw(6) << r.efC << ' '
                << std::setw(6) << r.efS << ' '
                << std::setw(10) << r.ivf_nlist << ' '
                << std::setw(10) << r.ivf_nprobe << ' '
                << std::setw(6) << r.pq_m << ' '
                << std::setw(10) << r.pq_nbits << ' '
                << std::setw(10) << r.rabitq_qb << ' '
                << std::setw(12) << r.rabitq_centered << ' '
                << std::setw(10) << r.refine_k << ' '
                << std::setw(12) << (r.refine_type.empty() ? "NA" : r.refine_type) << ' '
                << std::setw(10)
                << ((r.method == std::string("HNSW-SQ") ||
                     r.method == std::string("IVF-SQ"))
                        ? PQBenchmark::qtypeName(r.qtype)
                        : "NA")
                << ' ' << std::setw(12) << std::fixed << std::setprecision(2)
                << r.train_time << ' ' << std::setw(10) << std::fixed
                << std::setprecision(2) << r.add_time << ' ' << std::setw(12)
                << std::fixed << std::setprecision(2) << r.build_time << ' '
                << std::setw(14) << std::fixed << std::setprecision(2)
                << r.search_time_ms << ' ' << std::setw(10) << std::fixed
                << std::setprecision(3) << r.recall_1 << ' ' << std::setw(10)
                << std::fixed << std::setprecision(3) << r.recall_5 << ' '
                << std::setw(10) << std::fixed << std::setprecision(3)
                << r.recall_10 << ' ' << std::setw(14) << std::fixed
                << std::setprecision(1) << r.mbs_on_disk;
      // compression ratio: not applicable for HNSW-Flat
      {
        std::ostringstream cr;
        if (r.method == "HNSW-Flat" || r.method == "IVF-Flat") {
          cr << "NA";
        } else {
          cr << std::fixed << std::setprecision(1) << r.compression_ratio;
        }
        std::cout << ' ' << std::setw(13) << cr.str() << std::endl;
      }
    }
  }

  void saveResults(const std::vector<TestResult> &results,
                   const std::string &path) {
    std::ofstream csv_file(path);
    if (csv_file.is_open()) {
      csv_file << "method,dim,nb,nq,k,hnsw_M,efC,efS,ivf_nlist,ivf_nprobe,pq_m,"
                  "pq_nbits,rabitq_qb,rabitq_centered,refine_k,refine_type,"
                  "qtype,train_"
                  "time,add_time,build_time,search_time_ms,recall_at_1,"
                  "recall_at_5,recall_at_k,mbs_on_disk,compression_ratio\n";
      for (const auto &r : results) {
        csv_file << r.method << "," << r.dim << "," << r.nb << "," << r.nq
                 << "," << r.k << "," << r.hnsw_M << "," << r.efC << ","
                 << r.efS << "," << r.ivf_nlist << "," << r.ivf_nprobe << ","
                 << r.pq_m << "," << r.pq_nbits << "," << r.rabitq_qb << ","
                 << r.rabitq_centered << "," << r.refine_k << ","
                 << (r.refine_type.empty() ? std::string("NA") : r.refine_type)
                 << ","
                 << ((r.method == std::string("HNSW-SQ") ||
                      r.method == std::string("IVF-SQ"))
                         ? qtypeName(r.qtype)
                         : std::string("NA"))
                 << "," << r.train_time << "," << r.add_time << ","
                 << r.build_time << "," << r.search_time_ms << "," << r.recall_1
                 << "," << r.recall_5 << "," << r.recall_10 << ","
                 << r.mbs_on_disk << ",";
        if (r.method == "HNSW-Flat" || r.method == "IVF-Flat")
          csv_file << "NA";
        else
          csv_file << r.compression_ratio;
        csv_file << "\n";
      }
      csv_file.close();
      std::cout << "\n💾 结果已保存到: " << path << std::endl;
    } else {
      std::cerr << "❌ 无法创建CSV文件" << std::endl;
    }
  }
};

static int parseQType(const std::string &s) {
  using QT = faiss::ScalarQuantizer::QuantizerType;
  if (s == "QT_8bit")
    return (int)QT::QT_8bit;
  if (s == "QT_4bit")
    return (int)QT::QT_4bit;
  if (s == "QT_8bit_uniform")
    return (int)QT::QT_8bit_uniform;
  if (s == "QT_fp16")
    return (int)QT::QT_fp16;
  // fallback: try int
  try {
    return std::stoi(s);
  } catch (...) {
    return (int)QT::QT_8bit;
  }
}

static std::vector<std::string> splitCSV(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty())
      out.push_back(item);
  }
  return out;
}

static std::vector<int> toIntList(const std::string &s) {
  std::vector<int> v;
  auto parts = splitCSV(s);
  v.reserve(parts.size());
  for (auto &p : parts) {
    try {
      v.push_back(std::stoi(p));
    } catch (...) {
    }
  }
  return v;
}

static std::vector<double> toDoubleList(const std::string &s) {
  std::vector<double> v;
  auto parts = splitCSV(s);
  v.reserve(parts.size());
  for (auto &p : parts) {
    try {
      v.push_back(std::stod(p));
    } catch (...) {
    }
  }
  return v;
}

int main(int argc, char **argv) {
  Options opt;
  // Naive CLI parsing
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int &dst) {
      if (i + 1 < argc)
        dst = std::stoi(argv[++i]);
    };
    auto nexts = [&](std::string &dst) {
      if (i + 1 < argc)
        dst = argv[++i];
    };
    auto handleListOrSingleString = [&](std::vector<std::string> &lst,
                                        std::string &single) {
      std::string s;
      nexts(s);
      if (s.find(',') != std::string::npos)
        lst = splitCSV(s);
      else if (!s.empty())
        single = s;
    };
    auto handleListOrSingleInt = [&](std::vector<int> &lst, int &single) {
      std::string s;
      nexts(s);
      if (s.find(',') != std::string::npos)
        lst = toIntList(s);
      else {
        try {
          single = std::stoi(s);
        } catch (...) {
        }
      }
    };
    if (a == "--dim") {
      handleListOrSingleInt(opt.dim_list, opt.dim);
    } else if (a == "--nb") {
      handleListOrSingleInt(opt.nb_list, opt.nb);
    } else if (a == "--nq") {
      handleListOrSingleInt(opt.nq_list, opt.nq);
    } else if (a == "--k") {
      handleListOrSingleInt(opt.k_list, opt.k);
    } else if (a == "--hnsw-M") {
      handleListOrSingleInt(opt.hnsw_M_list, opt.hnsw_M);
    } else if (a == "--efC") {
      handleListOrSingleInt(opt.efC_list, opt.efC);
    } else if (a == "--efS") {
      handleListOrSingleInt(opt.efS_list, opt.efS);
    } else if (a == "--pq-m") {
      handleListOrSingleInt(opt.pq_m_list, opt.pq_m);
    } else if (a == "--pq-nbits") {
      handleListOrSingleInt(opt.pq_nbits_list, opt.pq_nbits);
    } else if (a == "--pq-train-ratio") {
      std::string s;
      nexts(s);
      if (s.find(',') != std::string::npos) {
        opt.pq_train_ratio_list = toDoubleList(s);
      } else {
        try {
          opt.pq_train_ratio = std::stod(s);
        } catch (...) {
        }
        if (opt.pq_train_ratio < 0.0)
          opt.pq_train_ratio = 0.0;
        if (opt.pq_train_ratio > 1.0)
          opt.pq_train_ratio = 1.0;
      }
    // } else if (a == "--sq-qtype") {  // removed - now specified in --which
    //   std::string s;
    //   nexts(s);
    //   if (s.find(',') != std::string::npos) {
    //     auto parts = splitCSV(s);
    //     for (auto &p : parts)
    //       opt.sq_qtype_list.push_back(parseQType(p));
    //   } else {
    //     opt.sq_qtype = parseQType(s);
    //   }
    } else if (a == "--ivf-nlist") {
      handleListOrSingleInt(opt.ivf_nlist_list, opt.ivf_nlist);
    } else if (a == "--ivf-nprobe") {
      handleListOrSingleInt(opt.ivf_nprobe_list, opt.ivf_nprobe);
    } else if (a == "--rabitq-qb") {
      handleListOrSingleInt(opt.rabitq_qb_list, opt.rabitq_qb);
    } else if (a == "--rabitq-centered") {
      std::string s;
      nexts(s);
      if (s.find(',') != std::string::npos) {
        opt.rabitq_centered_list = toIntList(s);
        for (int &v : opt.rabitq_centered_list) {
          v = v ? 1 : 0;
        }
      } else if (!s.empty()) {
        try {
          opt.rabitq_centered = (std::stoi(s) != 0);
        } catch (...) {
          std::string lowered = s;
          std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                         [](unsigned char c) { return std::tolower(c); });
          if (lowered == "true" || lowered == "t" || lowered == "yes" ||
              lowered == "y") {
            opt.rabitq_centered = true;
          } else if (lowered == "false" || lowered == "f" || lowered == "no" ||
                     lowered == "n") {
            opt.rabitq_centered = false;
          }
        }
      } else {
        // flag form defaults to true when no explicit value provided
        opt.rabitq_centered = true;
      }
    } else if (a == "--rbq-refine-type") {
      handleListOrSingleString(opt.rbq_refine_type_list, opt.rbq_refine_type);
    } else if (a == "--rbq-refine-k") {
      handleListOrSingleInt(opt.rbq_refine_k_list, opt.rbq_refine_k);
    } else if (a == "--which") {
      std::string s;
      nexts(s);
      opt.which = splitCSV(s);
    } else if (a == "--out-csv")
      nexts(opt.out_csv);
    else if (a == "--save-data-dir")
      nexts(opt.save_data_dir);
    else if (a == "--load-data-dir")
      nexts(opt.load_data_dir);
    else if (a == "--transpose-centroid") {
      opt.transpose_centroid = true;
    } else if (a == "--no-transpose-centroid") {
      opt.transpose_centroid = false; // explicit disable
    } else if (a == "--vecml-base-path") {
      nexts(opt.vecml_base_path);
    } else if (a == "--vecml-license") {
      nexts(opt.vecml_license_path);
    } else if (a == "-h" || a == "--help") {
    std::cout
      << "Usage: ./bench [options]\n"
       "  --dim INT[,INT...]           vector dim list (default 128)\n"
             "  --nb INT[,INT...]            database size list (default "
             "20000)\n"
             "  --nq INT[,INT...]            queries count list (default 500)\n"
             "  --k INT[,INT...]             top-k list (default 10)\n"
             "  --hnsw-M INT[,INT...]        HNSW M list (default 32)\n"
             "  --efC INT[,INT...]           HNSW efConstruction list (default "
             "200)\n"
             "  --efS INT[,INT...]           HNSW efSearch list (default 64)\n"
             "  --pq-m INT[,INT...]          PQ m list for HNSW-PQ (default "
             "16)\n"
             "  --pq-nbits INT[,INT...]      PQ nbits list for HNSW-PQ "
             "(default 8)\n"
             "  --pq-train-ratio F[,F...]    PQ train ratio list in [0,1] "
             "(default 1.0)\n"
             "  --ivf-nlist INT[,INT...]     IVF nlist list (default 256)\n"
             "  --ivf-nprobe INT[,INT...]    IVF nprobe list (default 8)\n"
             "  --rabitq-qb INT[,INT...]     RaBitQ qb bits for query (default "
             "8)\n"
             "  --rabitq-centered 0|1[,0|1...] RaBitQ centered query flag "
             "(default 0)\n"
             "  --rbq-refine-type STR[,STR...]  refine type for IVFRaBitQ "
             "refine (flat|sq8|sq4|fp16|bf16) (default flat)\n"
             "  --rbq-refine-k INT[,INT...]     refine k-factor (>=1), "
             "multiplier of k for first-stage candidates (default 2)\n"
             "  --which STR                  comma list: "
             "all|hnsw_flat|hnsw_sq4|hnsw_sq8|hnsw_pq|hnsw_rabitq|ivf_flat|ivf_sq4|ivf_sq8|ivf_pq|"
             "ivf_rbq|rbq_refine|ivf_rbq_refine|rabitq "
             "(default all)\n"
             "  --out-csv PATH      output csv path (default "
             "hnsw_index_types_results.csv)\n"
             "  --save-data-dir PATH         save test data (database, queries, groundtruth) "
             "to specified directory\n"
             "  --load-data-dir PATH         load test data from specified directory "
             "(if files exist, skips generation)\n"
             "  --transpose-centroid         enable PQ transposed centroids "
             "(default off)\n"
             "  --no-transpose-centroid      disable PQ transposed centroids "
             "(override)\n"
             "  --vecml-base-path PATH         path to VecML SDK directory (contains lib and headers)\n"
             "  --vecml-license PATH           path to VecML license file (optional, default: license.txt)\n";
      return 0;
    }
  }

  try {
    PQBenchmark bench(opt.dim, opt.nb, opt.nq, opt.k, opt.transpose_centroid, opt.save_data_dir, opt.load_data_dir);
    auto results = bench.runIndexTypeBenchmarks(opt);

    // If VecML is requested (either via --which=vecml or by providing a
    // --vecml-base-path) run the VecML test using the C shim. This keeps the
    // command-line simple: either add "vecml" to --which or provide
    // --vecml-base-path.
    bool want_vecml = false;
    for (const auto &w : opt.which) {
      if (w == "vecml") {
        want_vecml = true;
        break;
      }
    }
    if (!opt.vecml_base_path.empty()) {
      want_vecml = true;
    }
#ifdef HAVE_VECML
    if (want_vecml) {
      auto vr = bench.testVecML(opt.vecml_base_path, opt.vecml_license_path);
      if (vr.build_time >= 0) {
        results.push_back(vr);
      } else {
        std::cerr << "VecML test failed (see messages above)" << std::endl;
      }
    }
#else
    if (want_vecml) {
      std::cerr << "VecML support not compiled in (HAVE_VECML not defined)." << std::endl;
    }
#endif
    bench.printAnalysis(results);
    bench.saveResults(results, opt.out_csv);
    std::cout << "\n✅ 分析完成！" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "❌ 错误: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
