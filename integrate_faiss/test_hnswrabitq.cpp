#include <faiss/IndexHNSW.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/MetricType.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr float kFloatEps = 1e-4f;

bool almost_equal(float a, float b) {
    const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= kFloatEps * scale;
}

bool labels_match(
        const std::vector<faiss::idx_t>& lhs,
        const std::vector<faiss::idx_t>& rhs,
        const char* context) {
    if (lhs == rhs) {
        return true;
    }
    std::cerr << "Label mismatch in " << context << std::endl;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            std::cerr << "  idx " << i << ": " << lhs[i] << " vs " << rhs[i]
                      << std::endl;
        }
    }
    return false;
}

bool distances_match(
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        const char* context) {
    bool ok = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!almost_equal(lhs[i], rhs[i])) {
            std::cerr << "Distance mismatch in " << context << " at " << i
                      << ": " << lhs[i] << " vs " << rhs[i] << std::endl;
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "Failed context: " << context << std::endl;
    }
    return ok;
}

} // namespace

int main() {
    constexpr int dim = 32;
    constexpr int nb = 512;
    constexpr int nq = 16;
    constexpr int k = 10;
    constexpr int M = 16;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> database(nb * dim);
    std::vector<float> queries(nq * dim);

    for (float& v : database) {
        v = dist(rng);
    }

    for (float& v : queries) {
        v = dist(rng);
    }

    faiss::IndexRaBitQ baseline(dim, faiss::METRIC_L2);
    baseline.train(nb, database.data());
    baseline.add(nb, database.data());

    faiss::IndexHNSWRaBitQ index(dim, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = nb;
    index.hnsw.efSearch = nb;
    index.train(nb, database.data());
    index.add(nb, database.data());

    std::vector<float> distances_hnsw(nq * k);
    std::vector<faiss::idx_t> labels_hnsw(nq * k);
    std::vector<float> distances_baseline(nq * k);
    std::vector<faiss::idx_t> labels_baseline(nq * k);

    index.search(nq, queries.data(), k, distances_hnsw.data(), labels_hnsw.data());
    baseline.search(
            nq,
            queries.data(),
            k,
            distances_baseline.data(),
            labels_baseline.data());

    bool ok = true;

    ok &= labels_match(labels_hnsw, labels_baseline, "default search labels");
    ok &= distances_match(
        distances_hnsw, distances_baseline, "default search distances");

    std::vector<float> distances_hnsw_q(nq * k);
    std::vector<faiss::idx_t> labels_hnsw_q(nq * k);
    std::vector<float> distances_baseline_q(nq * k);
    std::vector<faiss::idx_t> labels_baseline_q(nq * k);

    faiss::SearchParametersHNSWRaBitQ hnsw_params_q;
    hnsw_params_q.efSearch = nb;
    hnsw_params_q.qb = 4;
    hnsw_params_q.centered = false;

    faiss::RaBitQSearchParameters baseline_params_q;
    baseline_params_q.qb = 4;
    baseline_params_q.centered = false;

    index.search(
            nq,
            queries.data(),
            k,
            distances_hnsw_q.data(),
            labels_hnsw_q.data(),
            &hnsw_params_q);
    baseline.search(
            nq,
            queries.data(),
            k,
            distances_baseline_q.data(),
            labels_baseline_q.data(),
            &baseline_params_q);

    ok &= labels_match(labels_hnsw_q, labels_baseline_q, "qb=4 labels");
    ok &= distances_match(
        distances_hnsw_q, distances_baseline_q, "qb=4 distances");

    faiss::SearchParametersHNSWRaBitQ hnsw_params_centered;
    hnsw_params_centered.efSearch = nb;
    hnsw_params_centered.qb = 3;
    hnsw_params_centered.centered = true;

    faiss::RaBitQSearchParameters baseline_params_centered;
    baseline_params_centered.qb = 3;
    baseline_params_centered.centered = true;

    std::vector<float> distances_hnsw_c(nq * k);
    std::vector<faiss::idx_t> labels_hnsw_c(nq * k);
    std::vector<float> distances_baseline_c(nq * k);
    std::vector<faiss::idx_t> labels_baseline_c(nq * k);

    index.search(
            nq,
            queries.data(),
            k,
            distances_hnsw_c.data(),
            labels_hnsw_c.data(),
            &hnsw_params_centered);
    baseline.search(
            nq,
            queries.data(),
            k,
            distances_baseline_c.data(),
            labels_baseline_c.data(),
            &baseline_params_centered);

    ok &= labels_match(
        labels_hnsw_c, labels_baseline_c, "centered qb=3 labels");
    ok &= distances_match(
        distances_hnsw_c,
        distances_baseline_c,
        "centered qb=3 distances");

    if (!ok) {
    std::cerr << "HNSW-RaBitQ tests failed" << std::endl;
    return 1;
    }

    std::cout << "HNSW-RaBitQ tests passed" << std::endl;
    return 0;
}
