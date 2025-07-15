#include "faiss/IndexFlat.h"
#include <iostream>

int main() {
    // Test faiss
    faiss::IndexFlatL2 index(128); // Create a flat L2 index for 128-dimensional vectors
    float x[128] = {0}; // Example vector
    index.add(1, x); // Add the vector to the index
    long n = index.ntotal; // Get the number of vectors in the index
    std::cout << "Number of vectors in index: " << n << std::endl;

}