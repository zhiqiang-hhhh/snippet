#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    // Test HNSWPQ index
    int dim = 128;  // Higher dimensional vectors for PQ to be meaningful
    int nb = 10000; // Number of vectors to add
    int nq = 10;    // Number of queries
    
    std::cout << "=== Testing FAISS HNSWPQ Index ===" << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Number of vectors: " << nb << std::endl;
    std::cout << "Number of queries: " << nq << std::endl;
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    std::vector<float> database(nb * dim);
    std::vector<float> queries(nq * dim);
    
    for (int i = 0; i < nb * dim; i++) {
        database[i] = dist(rng);
    }
    
    for (int i = 0; i < nq * dim; i++) {
        queries[i] = dist(rng);
    }
    
    // HNSWPQ parameters
    int M = 16;          // Number of connections per node in HNSW graph
    int nbits = 8;       // Number of bits per PQ code
    int m = 16;          // Number of subquantizers (dim should be divisible by m)
    
    std::cout << "\nHNSWPQ Parameters:" << std::endl;
    std::cout << "M (HNSW connections): " << M << std::endl;
    std::cout << "m (PQ subquantizers): " << m << std::endl;
    std::cout << "nbits (bits per code): " << nbits << std::endl;
    
    // Create HNSWPQ index
    faiss::IndexHNSWPQ index(dim, m, nbits, M, faiss::METRIC_L2);
    
    std::cout << "\nIndex created successfully" << std::endl;
    std::cout << "Index is trained: " << (index.is_trained ? "Yes" : "No") << std::endl;
    
    // Train the index if needed
    if (!index.is_trained) {
        std::cout << "Training index..." << std::endl;
        auto start_train = std::chrono::high_resolution_clock::now();
        
        index.train(nb, database.data());
        
        auto end_train = std::chrono::high_resolution_clock::now();
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train);
        std::cout << "Training completed in " << train_time.count() << " ms" << std::endl;
    }
    
    // Add vectors to index
    std::cout << "\nAdding vectors to index..." << std::endl;
    auto start_add = std::chrono::high_resolution_clock::now();
    
    index.add(nb, database.data());
    
    auto end_add = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_add - start_add);
    
    std::cout << "Added " << index.ntotal << " vectors in " << add_time.count() << " ms" << std::endl;
    std::cout << "Index size: " << index.ntotal << " vectors" << std::endl;
    
    // Search parameters
    int k = 10;  // Number of nearest neighbors to find
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    // Test search
    std::cout << "\nSearching for " << k << " nearest neighbors..." << std::endl;
    auto start_search = std::chrono::high_resolution_clock::now();
    
    index.search(nq, queries.data(), k, distances.data(), labels.data());
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);
    
    std::cout << "Search completed in " << search_time.count() << " μs" << std::endl;
    std::cout << "Average time per query: " << search_time.count() / nq << " μs" << std::endl;
    
    // Display first few search results
    std::cout << "\nFirst 3 search results:" << std::endl;
    for (int i = 0; i < std::min(3, nq); i++) {
        std::cout << "Query " << i << ":" << std::endl;
        for (int j = 0; j < k; j++) {
            if (labels[i * k + j] != -1) {
                std::cout << "  Rank " << j + 1 << ": ID=" << labels[i * k + j] 
                         << ", Distance=" << distances[i * k + j] << std::endl;
            }
        }
    }
    
    // Test different search parameters
    std::cout << "\n=== Testing Different Search Parameters ===" << std::endl;
    
    // Test with different efSearch values
    std::vector<int> ef_values = {16, 32, 64, 128};
    
    for (int ef : ef_values) {
        index.hnsw.efSearch = ef;
        
        auto start = std::chrono::high_resolution_clock::now();
        index.search(nq, queries.data(), k, distances.data(), labels.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "efSearch=" << ef << ": " << time.count() / nq << " μs per query" << std::endl;
    }
    
    // Compare with IndexHNSWFlat for reference
    std::cout << "\n=== Comparing with IndexHNSWFlat ===" << std::endl;
    
    faiss::IndexHNSWFlat flat_index(dim, M, faiss::METRIC_L2);
    
    auto start_flat_add = std::chrono::high_resolution_clock::now();
    flat_index.add(nb, database.data());
    auto end_flat_add = std::chrono::high_resolution_clock::now();
    auto flat_add_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_flat_add - start_flat_add);
    
    auto start_flat_search = std::chrono::high_resolution_clock::now();
    flat_index.search(nq, queries.data(), k, distances.data(), labels.data());
    auto end_flat_search = std::chrono::high_resolution_clock::now();
    auto flat_search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_flat_search - start_flat_search);
    
    std::cout << "IndexHNSWFlat - Add time: " << flat_add_time.count() << " ms" << std::endl;
    std::cout << "IndexHNSWFlat - Search time: " << flat_search_time.count() / nq << " μs per query" << std::endl;
    
    // Memory usage estimation
    std::cout << "\n=== Memory Usage Estimation ===" << std::endl;
    size_t pq_memory = nb * m * nbits / 8;  // PQ codes
    size_t hnsw_memory = nb * M * sizeof(faiss::idx_t);  // HNSW graph (approximate)
    
    std::cout << "Estimated PQ codes memory: " << pq_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Estimated HNSW graph memory: " << hnsw_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Total estimated memory: " << (pq_memory + hnsw_memory) / 1024 / 1024 << " MB" << std::endl;
    
    size_t flat_memory = nb * dim * sizeof(float);
    std::cout << "IndexHNSWFlat memory: " << flat_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Memory compression ratio: " << (float)flat_memory / (pq_memory + hnsw_memory) << "x" << std::endl;
    
    return 0;
}