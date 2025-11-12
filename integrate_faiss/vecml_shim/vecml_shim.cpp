#include "vecml_shim/vecml_shim.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
  std::string
      base_path; // store base path provided at creation for disk-size queries
  // Fast index options
  bool use_fast_index = false;
  float fast_shrink_rate = 0.4f;
  int fast_max_samples = 100000;
  int fast_index_threads = 16;
};

extern "C" {

vecml_ctx_t vecml_create(const char *base_path, const char *license_path, bool fast_index) {
  try {
    // Clean up any existing test database (actually remove the directory so
    // we don't accidentally load an old index with mismatched settings).
    try {
      if (base_path && !std::string(base_path).empty()) {
        std::error_code ec;
        std::filesystem::remove_all(base_path, ec);
        if (ec) {
          std::cerr << "vecml_create: failed to remove existing base path "
                    << base_path << " error=" << ec.message() << "\n";
        } else {
          std::cerr << "vecml_create: removed existing base path " << base_path
                    << "\n";
        }
      } else {
        std::cerr << "vecml_create: base_path is null or empty\n";
      }
    } catch (const std::exception &e) {
      std::cerr << "vecml_create: exception while removing base path: "
                << e.what() << "\n";
    }
    VecMLHandle *h = new VecMLHandle();
    h->api = std::make_unique<fluffy::FluffyInterface>(
        std::string(base_path), std::string(license_path));
    h->base_path = base_path ? std::string(base_path) : std::string();
    // Initialize the underlying Fluffy/VecML instance and attach a default
    // index so subsequent add/search calls work without extra setup from the
    // caller.
    fluffy::ErrorCode ec = h->api->init();
    if (ec != fluffy::ErrorCode::Success) {
      std::cerr << "vecml_create error: init failed: " << static_cast<int>(ec)
                << std::endl;
      delete h;
      return nullptr;
    }

    // Choose a default index name. Don't rely on get_vector_dim() here because
    // no vectors have been added yet; attach_index will be invoked later from
    // vecml_add_data when the vector dimension is known (to ensure the index
    // is created with the correct dim and distance metric).
    std::string idx_name = "test_index";
    h->index_name = idx_name;
    return reinterpret_cast<vecml_ctx_t>(h);
  } catch (const std::exception &e) {
    std::cerr << "vecml_create error: " << e.what() << std::endl;
    return nullptr;
  }
}

int vecml_add_data(vecml_ctx_t ctx, const float *data, int n, int dim,
                   const long *ids) {
  if (!ctx)
    return -1;
  VecMLHandle *h = reinterpret_cast<VecMLHandle *>(ctx);
  try {
    // Ensure an index exists with the correct dimension and distance type.
    // If no index was attached earlier (or vector dim is unknown), attach
    // one now using the provided `dim` and Euclidean distance so results
    // match Faiss's L2 baseline.
    if (h->api->get_vector_dim() == 0) {
      // cast to int to match the SDK overload - dim_t is an SDK typedef
      fluffy::ErrorCode attach_ec = fluffy::ErrorCode::Success;
      if (h->use_fast_index) {
        // Use fast index (soil index)
        h->index_name = "fast_index";
        attach_ec = h->api->attach_soil_index(
            (int)dim, "dense", fluffy::DistanceFunctionType::Euclidean,
            h->fast_shrink_rate, h->index_name, h->fast_max_samples,
            h->fast_index_threads);
      } else {
        attach_ec = h->api->attach_index(
            (int)dim, "dense", fluffy::DistanceFunctionType::Euclidean,
            h->index_name, 16);
      }
      if (attach_ec != fluffy::ErrorCode::Success) {
        std::cerr << "vecml_add_data warning: attach_index returned "
                  << static_cast<int>(attach_ec) << std::endl;
        // continue: add_data_batch may create index implicitly, but we tried
        // to make behavior explicit for correct metric/dim.
      }
    }
    // Prepare batch of (string_id, unique_ptr<Vector>) as the SDK expects
    std::vector<std::pair<std::string, std::unique_ptr<fluffy::Vector>>> batch;
    batch.reserve(n);
    for (int i = 0; i < n; ++i) {
      std::vector<float> embedding;
      embedding.assign(data + (size_t)i * dim, data + (size_t)i * dim + dim);
      std::unique_ptr<fluffy::Vector> vec;
      fluffy::ErrorCode ec = h->api->build_vector_dense(embedding, vec);
      if (ec != fluffy::ErrorCode::Success || !vec) {
        std::cerr << "vecml_add_data: build_vector_dense failed for index " << i
                  << "\n";
        return -2;
      }
      long id = ids ? ids[i] : i;
      std::string sid = std::string("id_") + std::to_string(id);
      // set attribute id and keep the attribute string alive until
      // set_attribute copies it
      std::string id_str = std::to_string(id);
      vec->set_attribute("id", reinterpret_cast<const uint8_t *>(id_str.data()),
                         id_str.size());
      batch.emplace_back(sid, std::move(vec));
    }

    std::vector<fluffy::ErrorCode> ecs =
        h->api->add_data_batch(batch, /*threads=*/16);
    // check results
    for (const auto &ec : ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        std::cerr << "vecml_add_data: add_data_batch returned error code "
                  << static_cast<int>(ec) << std::endl;
        // continue but report failure
      }
    }
    // Flush to ensure data is persisted/visible for subsequent searches
    fluffy::ErrorCode flush_ec = h->api->flush();
    if (flush_ec != fluffy::ErrorCode::Success) {
      std::cerr << "vecml_add_data: flush returned "
                << static_cast<int>(flush_ec) << std::endl;
    } else {
      std::cerr << "vecml_add_data: flush successful\n";
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "vecml_add_data error: " << e.what() << std::endl;
    return -2;
  }
}

int vecml_search(vecml_ctx_t ctx, const float *queries, int nq, int dim, int k,
                 long *out_ids) {
  if (!ctx)
    return -1;
  VecMLHandle *h = reinterpret_cast<VecMLHandle *>(ctx);
  try {
    // Build query vectors and Query objects
    std::vector<fluffy::Query> queries_vec;
    std::vector<std::unique_ptr<fluffy::Vector>>
        qvecs_hold; // keep ownership alive
    queries_vec.reserve(nq);
    qvecs_hold.reserve(nq);
    for (int i = 0; i < nq; ++i) {
      std::vector<float> embedding;
      embedding.assign(queries + (size_t)i * dim,
                       queries + (size_t)i * dim + dim);
      std::unique_ptr<fluffy::Vector> vec;
      fluffy::ErrorCode ec = h->api->build_vector_dense(embedding, vec);
      if (ec != fluffy::ErrorCode::Success || !vec) {
        std::cerr << "vecml_search: build_vector_dense failed for query " << i
                  << "\n";
        return -2;
      }
      fluffy::Query q;
      q.top_k = k;
      q.vector = vec.get();
      // Explicitly request Euclidean (L2) similarity so results align with
      // the Faiss L2 baseline used by the benchmark.
      q.similarity_measure = fluffy::DistanceFunctionType::Euclidean;
      // default similarity_measure will be used
      queries_vec.push_back(q);
      qvecs_hold.push_back(std::move(vec));
    }

    // Use the same index name attached during init (fallback to "test_index")
    const std::string idx =
        h->index_name.empty() ? std::string("test_index") : h->index_name;

    // Perform single searches for each query and marshal results
    for (int qi = 0; qi < nq; ++qi) {
      fluffy::InterfaceQueryResults qr;
      fluffy::ErrorCode sec =
          h->api->search(queries_vec[qi], qr, idx, 0.3f, 1.0f);
      if (sec != fluffy::ErrorCode::Success) {
        std::cerr << "vecml_search: search() failed for query " << qi
                  << " code=" << static_cast<int>(sec) << std::endl;
        // mark no results for this query
        for (int t = 0; t < k; ++t)
          out_ids[(size_t)qi * k + t] = -1;
        continue;
      }
      size_t found = 0;
      for (size_t j = 0; j < qr.results.size() && found < (size_t)k; ++j) {
        const std::string &sid = qr.results[j].string_id;
        long id = -1;
        size_t p = sid.find_last_of('_');
        if (p != std::string::npos) {
          try {
            id = std::stol(sid.substr(p + 1));
          } catch (...) {
            id = -1;
          }
        }
        out_ids[(size_t)qi * k + found] = id;
        ++found;
      }
      for (size_t t = found; t < (size_t)k; ++t)
        out_ids[(size_t)qi * k + t] = -1;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "vecml_search error: " << e.what() << std::endl;
    return -3;
  }
}

void vecml_destroy(vecml_ctx_t ctx) {
  if (!ctx)
    return;
  VecMLHandle *h = reinterpret_cast<VecMLHandle *>(ctx);
  delete h;
}

} // extern C

extern "C" {

double vecml_get_disk_mb(vecml_ctx_t ctx) {
  if (!ctx)
    return -1.0;
  VecMLHandle *h = reinterpret_cast<VecMLHandle *>(ctx);
  try {
    if (h->base_path.empty())
      return -1.0;
    std::uintmax_t total = 0;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(
             h->base_path,
             std::filesystem::directory_options::skip_permission_denied, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
      if (ec)
        continue;
      const auto &entry = *it;
      if (entry.is_regular_file(ec)) {
        std::error_code fec;
        auto sz = std::filesystem::file_size(entry.path(), fec);
        if (!fec)
          total += sz;
      }
    }
    double mb = static_cast<double>(total) / (1024.0 * 1024.0);
    return mb;
  } catch (...) {
    return -1.0;
  }
}

} // extern C

extern "C" {

int vecml_enable_fast_index(vecml_ctx_t ctx, float shrink_rate,
                            int max_num_samples, int num_threads) {
  if (!ctx)
    return -1;
  VecMLHandle *h = reinterpret_cast<VecMLHandle *>(ctx);
  if (shrink_rate <= 0.0f || shrink_rate > 1.0f)
    shrink_rate = 0.4f;
  if (max_num_samples <= 0)
    max_num_samples = 100000;
  if (num_threads <= 0)
    num_threads = 1;
  h->use_fast_index = true;
  h->fast_shrink_rate = shrink_rate;
  h->fast_max_samples = max_num_samples;
  h->fast_index_threads = num_threads;
  return 0;
}

} // extern C
