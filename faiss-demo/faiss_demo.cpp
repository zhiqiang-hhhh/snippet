#include "faiss/impl/AuxIndexStructures.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <iostream>


static void accumulate(double x, double y, double& sum) { sum += (x - y) * (x - y); }
static double finalize(double sum) { return sqrt(sum); }

int main() {
    // // Help me to compute l2 distance by using faiss
    // int d = 128;
    // int n = 10;
    // faiss::IndexFlatL2 index(d);
    // float* x = new float[d * n];
    // float* y = new float[d * n];
    // for (int i = 0; i < d * n; i++) {
    //     x[i] = static_cast<float>(i);
    //     y[i] = static_cast<float>(i + i / 2);
    // }
    // std::cout << "IndexFlatL2 adding "<< n << " vectors" << std::endl;
    // index.add(n, x);
    // std::cout << "IndexFlatL2 searching "<< n << " vectors" << std::endl;
    // faiss::idx_t* labels = new faiss::idx_t[n];
    // float* distances = new float[n];
    // index.search(n, y, 1, distances, labels);
    // for (int i = 0; i < n; i++) {
    //     std::cout << "Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;              
    // }

    // std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(128, 16);
    // hnsw_index->add(n, x);
    // hnsw_index->search(n, y, 1, distances, labels);  // Changed from n to 1 neighbors
    // for (int i = 0; i < n; i++) {
    //     std::cout << "HNSW Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;
    // }

    // faiss::write_index(hnsw_index.get(), "/tmp/hnsw.index");

    // faiss::IndexHNSWFlat* hnsw_index_read = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("/tmp/hnsw.index"));
    // if (hnsw_index_read) {
    //     std::cout << "Successfully read HNSW index from file." << std::endl;
    // } else {
    //     std::cerr << "Failed to read HNSW index from file." << std::endl;
    // }

    // hnsw_index_read->search(n, y, 1, distances, labels);  // Changed from n to 1 neighbors
    // for (int i = 0; i < n; i++) {
    //     std::cout << "HNSW Read Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;
    // }

    // Test setup
    int dim2 = 10;
    int n2 = 10;
    /*
    [0, 1, 2, ..., 9]
    [1, 2, 3, ..., 10]
    [2, 3, 4, ..., 11]
    ...
    [9, 10, 11, ..., 18]
    */
    std::vector<float> data_vectors(n2 * dim2);
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < dim2; ++j) {
            data_vectors[i * dim2 + j] = static_cast<float>(i + j);
        }
    }

    // Create Faiss index
    faiss::IndexFlatL2 index2(dim2);
    index2.add(n2, data_vectors.data());

    // Perform range search
    std::vector<float> query_vector;
    // [0, 1, 2, ..., 9]
    for (int j = 0; j < dim2; ++j) {
        query_vector.push_back(static_cast<float>(j));
    }
    
    std::cout << "Range search ground truth:" << std::endl;
    std::vector<float> ground_truth;
    for (size_t i = 0; i < data_vectors.size(); i += dim2) {
        double sum = 0;
        for (size_t j = 0; j < dim2; ++j) {
            accumulate(data_vectors[i + j], query_vector[j], sum);
        }
        ground_truth.push_back(finalize(sum));
    }

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        std::cout << "Distance to vector " << i << ": " << ground_truth[i] << std::endl;
    }
    std::cout << "Range search result:" << std::endl;

    faiss::RangeSearchResult result(1);
    float radius = 15.0f;
    std::cout << "radius: " << radius << std::endl;
    index2.range_search(2, query_vector.data(), radius*radius, &result);

    for (size_t i = 0; i < result.nq; ++i) {
        std::cout << "Query " << i << ":\n";
        std::cout << "Lims: ";
        std::cout << result.lims[i] << " " << result.lims[i + 1] << std::endl;
        size_t begin = result.lims[i];
        size_t end = result.lims[i + 1];
        std::cout << "Labels: ";
        for (size_t j = begin; j < end; ++j) {
            std::cout << result.labels[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Distances: ";
        for (size_t j = begin; j < end; ++j) {
            std::cout << result.distances[j] << " ";
        }

        std::cout << "\nDistance after sqrt:\n" << std::endl;
        for (size_t j = begin; j < end; ++j) {
            std::cout << sqrt(result.distances[j]) << " ";
        }

        std::cout << std::endl;
        std::cout << "------------------------" << std::endl;
    }
}