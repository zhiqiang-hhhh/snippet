#include "faiss/IndexFlat.h"
#include "faiss/IndexHNSW.h"
#include <iostream>

int main() {
    // Help me to compute l2 distance by using faiss
    int d = 128;
    int n = 10;
    faiss::IndexFlatL2 index(d);
    float* x = new float[d * n];
    float* y = new float[d * n];
    for (int i = 0; i < d * n; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i + 1);
    }
    std::cout << "IndexFlatL2 adding "<< n << " vectors" << std::endl;
    index.add(n, x);
    std::cout << "IndexFlatL2 searching "<< n << " vectors" << std::endl;
    faiss::idx_t* labels = new faiss::idx_t[n];
    float* distances = new float[n];
    index.search(n, y, 1, distances, labels);
    for (int i = 0; i < n; i++) {
        std::cout << "Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;              
    }

    std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(128, 16);
    hnsw_index->add(n, x);
    hnsw_index->search(n, y, 1, distances, labels);  // Changed from n to 1 neighbors
    for (int i = 0; i < n; i++) {
        std::cout << "HNSW Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;
    }

    delete[] x;
    delete[] y;
    delete[] labels;
    delete[] distances;
}