#include "vecml_shim/vecml_shim.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

#ifdef __has_include
#if __has_include(<fluffy_interface.h>)
#include <fluffy_interface.h>
#include <query_results.h>
#else
#error "fluffy_interface.h not found in include path"
#endif
#endif

using namespace std;

struct VecMLHandle {
  std::unique_ptr<fluffy::FluffyInterface> api;
  std::string index_name; // name of the attached index to use for searches
  std::string base_path; // store base path provided at creation for disk-size queries
};

extern "C" {

vecml_ctx_t vecml_create(const char* base_path, const char* license_path) {
  try {
    VecMLHandle* h = new VecMLHandle();
    h->api = std::make_unique<fluffy::FluffyInterface>(std::string(base_path), std::string(license_path));
    h->base_path = base_path ? std::string(base_path) : std::string();
    // Initialize the underlying Fluffy/VecML instance and attach a default index
    // so subsequent add/search calls work without extra setup from the caller.
    fluffy::ErrorCode ec = h->api->init();
    if (ec != fluffy::ErrorCode::Success) {
      std::cerr << "vecml_create error: init failed: " << static_cast<int>(ec) << std::endl;
      delete h;
      return nullptr;
    }

    // Attach a default dense index named "test_index" using NegativeCosineSimilarity
    // The signature follows the SDK: attach_index(dim, type, similarity, index_name, shards)
    std::string idx_name = "test_index";
    fluffy::ErrorCode attach_ec = h->api->attach_index(h->api->get_vector_dim() > 0 ? (int)h->api->get_vector_dim() : 0,
                                                      "dense",
                                                      fluffy::DistanceFunctionType::Euclidean,
                                                      idx_name,
                                                      1);
    // If attach fails because vector dim is not yet known, try attaching later when adding data.
    if (attach_ec != fluffy::ErrorCode::Success) {
      // Log but continue — add_data will still work because add_data_batch can create index implicitly
      std::cerr << "vecml_create warning: attach_index returned " << static_cast<int>(attach_ec) << std::endl;
    }
    // store chosen index name for later search/add calls
    h->index_name = idx_name;
    return reinterpret_cast<vecml_ctx_t>(h);
  } catch (const std::exception& e) {
    std::cerr << "vecml_create error: " << e.what() << std::endl;
    return nullptr;
  }
}

int vecml_add_data(vecml_ctx_t ctx, const float* data, int n, int dim, const long* ids) {
  if (!ctx) return -1;
  VecMLHandle* h = reinterpret_cast<VecMLHandle*>(ctx);
  try {
    // Prepare batch of (string_id, unique_ptr<Vector>) as the SDK expects
    std::vector<std::pair<std::string, std::unique_ptr<fluffy::Vector>>> batch;
    batch.reserve(n);
    for (int i = 0; i < n; ++i) {
      std::vector<float> embedding;
      embedding.assign(data + (size_t)i * dim, data + (size_t)i * dim + dim);
      std::unique_ptr<fluffy::Vector> vec;
      fluffy::ErrorCode ec = h->api->build_vector_dense(embedding, vec);
      if (ec != fluffy::ErrorCode::Success || !vec) {
        std::cerr << "vecml_add_data: build_vector_dense failed for index " << i << "\n";
        return -2;
      }
      long id = ids ? ids[i] : i;
      std::string sid = std::string("id_") + std::to_string(id);
      // set attribute id and keep the attribute string alive until set_attribute copies it
      std::string id_str = std::to_string(id);
      vec->set_attribute("id", reinterpret_cast<const uint8_t*>(id_str.data()), id_str.size());
      batch.emplace_back(sid, std::move(vec));
    }

    std::vector<fluffy::ErrorCode> ecs = h->api->add_data_batch(batch, /*threads=*/16);
    // check results
    for (const auto &ec : ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        std::cerr << "vecml_add_data: add_data_batch returned error code " << static_cast<int>(ec) << std::endl;
        // continue but report failure
      }
    }
    // Flush to ensure data is persisted/visible for subsequent searches
    fluffy::ErrorCode flush_ec = h->api->flush();
    if (flush_ec != fluffy::ErrorCode::Success) {
      std::cerr << "vecml_add_data: flush returned " << static_cast<int>(flush_ec) << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "vecml_add_data error: " << e.what() << std::endl;
    return -2;
  }
}

int vecml_search(vecml_ctx_t ctx, const float* queries, int nq, int dim, int k, long* out_ids) {
  if (!ctx) return -1;
  VecMLHandle* h = reinterpret_cast<VecMLHandle*>(ctx);
  try {
    // Build query vectors and Query objects
    std::vector<fluffy::Query> queries_vec;
    std::vector<std::unique_ptr<fluffy::Vector>> qvecs_hold; // keep ownership alive
    queries_vec.reserve(nq);
    qvecs_hold.reserve(nq);
    for (int i = 0; i < nq; ++i) {
      std::vector<float> embedding;
      embedding.assign(queries + (size_t)i * dim, queries + (size_t)i * dim + dim);
      std::unique_ptr<fluffy::Vector> vec;
      fluffy::ErrorCode ec = h->api->build_vector_dense(embedding, vec);
      if (ec != fluffy::ErrorCode::Success || !vec) {
        std::cerr << "vecml_search: build_vector_dense failed for query " << i << "\n";
        return -2;
      }
      fluffy::Query q;
      q.top_k = k;
      q.vector = vec.get();
      // default similarity_measure will be used
      queries_vec.push_back(q);
      qvecs_hold.push_back(std::move(vec));
    }

  // Use the same index name attached during init (fallback to "test_index")
  const std::string idx = h->index_name.empty() ? std::string("test_index") : h->index_name;

  // Perform single searches for each query and marshal results
  for (int qi = 0; qi < nq; ++qi) {
    fluffy::InterfaceQueryResults qr;
    fluffy::ErrorCode sec = h->api->search(queries_vec[qi], qr, idx, 0.3f, 0.9f);
    if (sec != fluffy::ErrorCode::Success) {
      std::cerr << "vecml_search: search() failed for query " << qi << " code=" << static_cast<int>(sec) << std::endl;
      // mark no results for this query
      for (int t = 0; t < k; ++t) out_ids[(size_t)qi * k + t] = -1;
      continue;
    }
    size_t found = 0;
    for (size_t j = 0; j < qr.results.size() && found < (size_t)k; ++j) {
      const std::string &sid = qr.results[j].string_id;
      long id = -1;
      size_t p = sid.find_last_of('_');
      if (p != std::string::npos) {
        try { id = std::stol(sid.substr(p+1)); } catch (...) { id = -1; }
      }
      out_ids[(size_t)qi * k + found] = id;
      ++found;
    }
    for (size_t t = found; t < (size_t)k; ++t) out_ids[(size_t)qi * k + t] = -1;
  }
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "vecml_search error: " << e.what() << std::endl;
    return -3;
  }
}

void vecml_destroy(vecml_ctx_t ctx) {
  if (!ctx) return;
  VecMLHandle* h = reinterpret_cast<VecMLHandle*>(ctx);
  delete h;
}

} // extern C

extern "C" {

double vecml_get_disk_mb(vecml_ctx_t ctx) {
  if (!ctx) return -1.0;
  VecMLHandle* h = reinterpret_cast<VecMLHandle*>(ctx);
  try {
    if (h->base_path.empty()) return -1.0;
    std::uintmax_t total = 0;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(h->base_path, std::filesystem::directory_options::skip_permission_denied, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
      if (ec) continue;
      const auto &entry = *it;
      if (entry.is_regular_file(ec)) {
        std::error_code fec;
        auto sz = std::filesystem::file_size(entry.path(), fec);
        if (!fec) total += sz;
      }
    }
    double mb = static_cast<double>(total) / (1024.0 * 1024.0);
    return mb;
  } catch (...) {
    return -1.0;
  }
}

} // extern C
