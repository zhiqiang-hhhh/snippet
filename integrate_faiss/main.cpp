#include "faiss/IndexHNSW.h"
#include "faiss/MetricType.h"
#include "faiss/impl/AuxIndexStructures.h"
#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>

int main() {
    // Test faiss index merge
    int dim = 4;  // 4-dimensional vectors
    
    // Create first index
    faiss::IndexHNSWFlat index1(dim, 16, faiss::METRIC_L2);
    
    // Add vectors to first index
    std::vector<float> vectors1 = {
        1.0, 2.0, 3.0, 4.0,    // vector 0
        5.0, 6.0, 7.0, 8.0,    // vector 1
        9.0, 10.0, 11.0, 12.0  // vector 2
    };
    index1.add(3, vectors1.data());
    
    // Create second index
    faiss::IndexHNSWFlat index2(dim, 16, faiss::METRIC_L2);
    
    // Add vectors to second index
    std::vector<float> vectors2 = {
        13.0, 14.0, 15.0, 16.0,  // vector 0
        17.0, 18.0, 19.0, 20.0,  // vector 1
        21.0, 22.0, 23.0, 24.0   // vector 2
    };
    index2.add(3, vectors2.data());
    
    std::cout << "Before merge:" << std::endl;
    std::cout << "Index1 size: " << index1.ntotal << std::endl;
    std::cout << "Index2 size: " << index2.ntotal << std::endl;
    
    // Merge index2 into index1
    try {
        index1.merge_from(index2, 0);  // 0 means add new IDs starting from index1.ntotal
    } catch (faiss::FaissException& e) {
        std::cerr << "Error during merge: " << e.what() << std::endl;
    }
    
    
    std::cout << "\nAfter merge:" << std::endl;
    std::cout << "Index1 size: " << index1.ntotal << std::endl;
    
    // Test search on merged index
    std::vector<float> query = {1.5, 2.5, 3.5, 4.5};
    int k = 5;
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    
    index1.search(1, query.data(), k, distances.data(), labels.data());
    
    std::cout << "\nSearch results on merged index:" << std::endl;
    std::cout << "Query: [" << query[0] << ", " << query[1] << ", " << query[2] << ", " << query[3] << "]" << std::endl;
    for (int i = 0; i < k && labels[i] != -1; i++) {
        std::cout << "Label: " << labels[i] << ", Distance: " << distances[i] << std::endl;
    }
    
    // Alternative: Test with IndexFlat for simpler merge verification
    std::cout << "\n--- Testing with IndexFlat ---" << std::endl;
    
    faiss::IndexFlat flat_index1(dim, faiss::METRIC_L2);
    faiss::IndexFlat flat_index2(dim, faiss::METRIC_L2);
    
    flat_index1.add(3, vectors1.data());
    flat_index2.add(3, vectors2.data());
    
    std::cout << "Before merge - Flat Index1 size: " << flat_index1.ntotal << std::endl;
    std::cout << "Before merge - Flat Index2 size: " << flat_index2.ntotal << std::endl;
    
    // Merge flat indices
    flat_index1.merge_from(flat_index2, 0);
    
    std::cout << "After merge - Flat Index1 size: " << flat_index1.ntotal << std::endl;
    
    // Search on merged flat index
    flat_index1.search(1, query.data(), k, distances.data(), labels.data());
    
    std::cout << "\nSearch results on merged flat index:" << std::endl;
    for (int i = 0; i < k && labels[i] != -1; i++) {
        std::cout << "Label: " << labels[i] << ", Distance: " << distances[i] << std::endl;
    }
    
    return 0;
}