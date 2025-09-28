#include <algorithm>
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
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

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

  // SQ (for HNSW-SQ)
  int sq_qtype = (int)faiss::ScalarQuantizer::QuantizerType::QT_8bit;
  std::vector<int> sq_qtype_list;

  // Which tests to run: all, hnsw_flat, hnsw_sq, hnsw_pq
  std::vector<std::string> which = {"all"};

  // IVF parameters (for IVF-Flat / IVF-PQ / IVF-SQ)
  int ivf_nlist = 256;
  int ivf_nprobe = 8;
  std::vector<int> ivf_nlist_list;
  std::vector<int> ivf_nprobe_list;

  // RaBitQ (for IVF-RaBitQ)
  int rabitq_qb = 8;            // query quantization bits
  bool rabitq_centered = false; // whether to center queries
  std::vector<int> rabitq_qb_list;

  // Output
  std::string out_csv = "hnsw_index_types_results.csv";

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

public:
  PQBenchmark(int dim = 128, int nb = 20000, int nq = 500, int k = 10,
              bool transpose_centroid = false)
      : dim(dim), nb(nb), nq(nq), k(k), transpose_centroid(transpose_centroid) {
    std::cout << "=== HNSW Index Types Benchmark (C++) ===" << std::endl;
    std::cout << "配置: dim=" << dim << ", nb=" << nb << ", nq=" << nq
              << ", k=" << k << std::endl;
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
    std::cout << "  测试 IndexHNSWSQ (" << qtypeName(qtype) << ", M=" << hnsw_m
              << ", efC=" << efC << ", efS=" << efS << ")..." << std::endl;
    TestResult r{};
    r.method = "HNSW-SQ";
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
    std::cout << "  测试 IndexIVFScalarQuantizer (" << qtypeName(qtype)
              << ", nlist=" << nlist << ", nprobe=" << nprobe << ")..."
              << std::endl;
    TestResult r{};
    r.method = "IVF-SQ";
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
      faiss::IndexFlatL2 coarse(dim);
      faiss::IndexIVFRaBitQ index(&coarse, dim, nlist, faiss::METRIC_L2);
      index.qb = (uint8_t)qb; // default query bits
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
      faiss::IVFRaBitQSearchParameters sp;
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
    } catch (const std::exception &e) {
      std::cerr << "    错误: " << e.what() << std::endl;
      r.build_time = -1;
    }
    return r;
  }

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
    auto sq_qtype_list = prepareIntList(opt.sq_qtype_list, opt.sq_qtype);
    auto ivf_nlist_list = prepareIntList(opt.ivf_nlist_list, opt.ivf_nlist);
    auto ivf_nprobe_list = prepareIntList(opt.ivf_nprobe_list, opt.ivf_nprobe);
    auto rabitq_qb_list = prepareIntList(opt.rabitq_qb_list, opt.rabitq_qb);

    // Iterate over dataset-level combinations first: dim, nb, nq, k
    for (int dim_v : dim_list) {
      for (int nb_v : nb_list) {
        for (int nq_v : nq_list) {
          for (int k_v : k_list) {
            // Rebuild benchmark data for each dataset combo
            PQBenchmark bench_local(dim_v, nb_v, nq_v, k_v,
                                    opt.transpose_centroid);
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
                  if (contains("hnsw_sq")) {
                    for (int qtype_v : sq_qtype_list) {
                      auto r =
                          bench_local.testHNSWSQ(qtype_v, hM, efC_v, efS_v);
                      if (r.build_time >= 0) {
                        r.dim = dim_v;
                        r.nb = nb_v;
                        r.nq = nq_v;
                        r.k = k_v;
                        results.push_back(r);
                      }
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
                if (contains("ivf_sq")) {
                  for (int qtype_v : sq_qtype_list) {
                    auto r = bench_local.testIVFSQ(nl_v, np_v, qtype_v);
                    if (r.build_time >= 0) {
                      r.dim = dim_v;
                      r.nb = nb_v;
                      r.nq = nq_v;
                      r.k = k_v;
                      results.push_back(r);
                    }
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
                if (contains("ivf_rabitq")) {
                  for (int qb_v : rabitq_qb_list) {
                    auto r = bench_local.testIVFRaBitQ(nl_v, np_v, qb_v,
                                                       opt.rabitq_centered);
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
    // Print header matching CSV columns and order
    std::cout << std::left << std::setw(12) << "method" << std::setw(6) << "dim"
              << std::setw(8) << "nb" << std::setw(6) << "nq" << std::setw(4)
              << "k" << std::setw(8) << "hnsw_M" << std::setw(6) << "efC"
              << std::setw(6) << "efS" << std::setw(10) << "ivf_nlist"
              << std::setw(10) << "ivf_nprobe" << std::setw(6) << "pq_m"
              << std::setw(10) << "pq_nbits" << std::setw(10) << "rabitq_qb"
              << std::setw(16) << "rabitq_centered" << std::setw(18) << "qtype"
              << std::setw(12) << "train_time" << std::setw(10) << "add_time"
              << std::setw(12) << "build_time" << std::setw(14)
              << "search_time_ms" << std::setw(12) << "recall_at_1"
              << std::setw(12) << "recall_at_5" << std::setw(14)
              << "recall_at_k" << std::setw(14) << "mbs_on_disk"
              << std::setw(16) << "compression_ratio" << std::endl;
    std::cout << std::string(12 + 6 + 8 + 6 + 4 + 8 + 6 + 6 + 10 + 10 + 6 + 10 +
                                 10 + 16 + 18 + 12 + 10 + 12 + 14 + 12 + 12 +
                                 14 + 14 + 16,
                             '-')
              << std::endl;

    for (const auto &r : results) {
      std::cout << std::left << std::setw(12) << r.method << std::setw(6)
                << r.dim << std::setw(8) << r.nb << std::setw(6) << r.nq
                << std::setw(4) << r.k << std::setw(8) << r.hnsw_M
                << std::setw(6) << r.efC << std::setw(6) << r.efS
                << std::setw(10) << r.ivf_nlist << std::setw(10) << r.ivf_nprobe
                << std::setw(6) << r.pq_m << std::setw(10) << r.pq_nbits
                << std::setw(10) << r.rabitq_qb << std::setw(16)
                << r.rabitq_centered << std::setw(18)
                << ((r.method == std::string("HNSW-SQ") ||
                     r.method == std::string("IVF-SQ"))
                        ? PQBenchmark::qtypeName(r.qtype)
                        : "NA")
                << std::setw(12) << std::fixed << std::setprecision(2)
                << r.train_time << std::setw(10) << std::fixed
                << std::setprecision(2) << r.add_time << std::setw(12)
                << std::fixed << std::setprecision(2) << r.build_time
                << std::setw(14) << std::fixed << std::setprecision(2)
                << r.search_time_ms << std::setw(12) << std::fixed
                << std::setprecision(3) << r.recall_1 << std::setw(12)
                << std::fixed << std::setprecision(3) << r.recall_5
                << std::setw(14) << std::fixed << std::setprecision(3)
                << r.recall_10 << std::setw(14) << std::fixed
                << std::setprecision(1) << r.mbs_on_disk;
      // compression ratio: not applicable for HNSW-Flat
      {
        std::ostringstream cr;
        if (r.method == "HNSW-Flat" || r.method == "IVF-Flat") {
          cr << "NA";
        } else {
          cr << std::fixed << std::setprecision(1) << r.compression_ratio;
        }
        std::cout << std::setw(16) << cr.str() << std::endl;
      }
    }
  }

  void saveResults(const std::vector<TestResult> &results,
                   const std::string &path) {
    std::ofstream csv_file(path);
    if (csv_file.is_open()) {
      csv_file << "method,dim,nb,nq,k,hnsw_M,efC,efS,ivf_nlist,ivf_nprobe,pq_m,"
                  "pq_nbits,rabitq_qb,rabitq_centered,qtype,train_"
                  "time,add_time,build_time,search_time_ms,recall_at_1,"
                  "recall_at_5,recall_at_k,mbs_on_disk,compression_ratio\n";
      for (const auto &r : results) {
        csv_file << r.method << "," << r.dim << "," << r.nb << "," << r.nq
                 << "," << r.k << "," << r.hnsw_M << "," << r.efC << ","
                 << r.efS << "," << r.ivf_nlist << "," << r.ivf_nprobe << ","
                 << r.pq_m << "," << r.pq_nbits << "," << r.rabitq_qb << ","
                 << r.rabitq_centered << ","
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
    } else if (a == "--sq-qtype") {
      std::string s;
      nexts(s);
      if (s.find(',') != std::string::npos) {
        auto parts = splitCSV(s);
        for (auto &p : parts)
          opt.sq_qtype_list.push_back(parseQType(p));
      } else {
        opt.sq_qtype = parseQType(s);
      }
    } else if (a == "--ivf-nlist") {
      handleListOrSingleInt(opt.ivf_nlist_list, opt.ivf_nlist);
    } else if (a == "--ivf-nprobe") {
      handleListOrSingleInt(opt.ivf_nprobe_list, opt.ivf_nprobe);
    } else if (a == "--rabitq-qb") {
      handleListOrSingleInt(opt.rabitq_qb_list, opt.rabitq_qb);
    } else if (a == "--rabitq-centered") {
      int v = 0;
      next(v);
      opt.rabitq_centered = (v != 0);
    } else if (a == "--which") {
      std::string s;
      nexts(s);
      opt.which = splitCSV(s);
    } else if (a == "--out-csv")
      nexts(opt.out_csv);
    else if (a == "--transpose-centroid") {
      opt.transpose_centroid = true;
    } else if (a == "--no-transpose-centroid") {
      opt.transpose_centroid = false; // explicit disable
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
             "  --sq-qtype STR[,STR...]      SQ qtype list "
             "(QT_8bit|QT_4bit|QT_8bit_uniform|QT_fp16)\n"
             "  --ivf-nlist INT[,INT...]     IVF nlist list (default 256)\n"
             "  --ivf-nprobe INT[,INT...]    IVF nprobe list (default 8)\n"
             "  --rabitq-qb INT[,INT...]     RaBitQ qb bits for query (default "
             "8)\n"
             "  --rabitq-centered 0|1        RaBitQ centered query flag "
             "(default 0)\n"
             "  --which STR                  comma list: "
             "all|hnsw_flat|hnsw_sq|hnsw_pq|ivf_flat|ivf_sq|ivf_pq|ivf_rabitq "
             "(default all)\n"
             "  --out-csv PATH      output csv path (default "
             "hnsw_index_types_results.csv)\n"
             "  --transpose-centroid         enable PQ transposed centroids "
             "(default off)\n"
             "  --no-transpose-centroid      disable PQ transposed centroids "
             "(override)\n";
      return 0;
    }
  }

  try {
    PQBenchmark bench(opt.dim, opt.nb, opt.nq, opt.k, opt.transpose_centroid);
    auto results = bench.runIndexTypeBenchmarks(opt);
    bench.printAnalysis(results);
    bench.saveResults(results, opt.out_csv);
    std::cout << "\n✅ 分析完成！" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "❌ 错误: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
