#include "faiss/IndexFlat.h"
#include "faiss/impl/DistanceComputer.h"
#include <iostream>

int main() {
    // Help me to compute l2 distance by using faiss
    faiss::IndexFlatL2 index(128);
    std::vector<float> x(128);
    std::vector<float> y(128);
    id_t n = 1000;
    index.add(n, x.data());
    std::vector<faiss::idx_t> labels(n);
    std::vector<float> distances(n);
    index.search(n, y.data(), n, distances.data(), labels.data());
    for (int i = 0; i < n; i++) {
        std::cout << "Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;              
    }
}