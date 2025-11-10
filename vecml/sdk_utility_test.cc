#include "fluffy_interface.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <cassert>

// Helper function to log test results
void log_test(const std::string& test_name, bool passed, const std::string& details = "") {
  std::string status = passed ? "PASS" : "FAIL";
  std::string message = "[" + status + "] " + test_name;
  if (!details.empty()) {
    message += " - " + details;
  }
  std::cout << message << std::endl;
}

// Helper function to create test vectors
inline std::unique_ptr<fluffy::Vector>
create_test_vector(fluffy::FluffyInterface& fluffyInterface,
                   int id,
                   int dim,
                   int bits_per_element = 32,
                   bool use_bit_vectors = false)
{
  // Create deterministic but varied float embedding in [0, 1]
  std::vector<float> embedding(dim);
  for (int i = 0; i < dim; ++i) {
    embedding[i] = std::sin(id * 0.1 + i * 0.01f) * 0.5f + 0.5f;
  }

  std::unique_ptr<fluffy::Vector> vector;
  if (use_bit_vectors && bits_per_element != 32) {
    std::vector<uint8_t> embedding_uint8(dim);
    for (int i = 0; i < dim; ++i) {
      float v = embedding[i];
      if (v < 0.f) v = 0.f;
      if (v > 1.f) v = 1.f;
      embedding_uint8[i] = static_cast<uint8_t>(v * 255.0f);
    }
    fluffyInterface.build_vector_bit(embedding_uint8, bits_per_element, vector);
  } else {
    fluffyInterface.build_vector_dense(embedding, vector);
  }

  // Set attributes
  const std::string str_id = std::to_string(id);
  vector->set_attribute("id",
                        reinterpret_cast<const uint8_t*>(str_id.data()),
                        str_id.size());

  const std::string category = (id % 2 == 0) ? "even" : "odd";
  vector->set_attribute("category",
                        reinterpret_cast<const uint8_t*>(category.data()),
                        category.size());

  return vector;
}

int main(int argc, char** argv) {
  std::cout << "=== Fluffy SDK Utility Test ===" << std::endl;
  
  bool all_tests_passed = true;
  std::stringstream test_log;

  try {
    // Test Configuration
    int dim = 384;
    std::string base_path = "./fluffy-test-db";
    std::string license_path = "license.txt"; // Update if license is required
    
    // Clean up any existing test database
    std::filesystem::remove_all(base_path);

    // =====================================================
    // Test 1: Initialization
    // =====================================================
    std::cout << "\n--- Test 1: Initialization ---" << std::endl;
    fluffy::FluffyInterface fluffyInterface(base_path, license_path);
    fluffy::ErrorCode init_ec = fluffyInterface.init();
    log_test("Initialization", init_ec == fluffy::ErrorCode::Success,
             "ErrorCode: " + std::to_string(static_cast<int>(init_ec)));
    if (init_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 2: Attach Index (full-precision)
    // =====================================================
    std::cout << "\n--- Test 2: Attach Index (full-precision) ---" << std::endl;
    fluffy::ErrorCode attach_ec = fluffyInterface.attach_index(
        dim, "dense", fluffy::DistanceFunctionType::NegativeCosineSimilarity, "test_index", 1);
    log_test("Attach Index", attach_ec == fluffy::ErrorCode::Success);
    if (attach_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 3: Build Vector Dense
    // =====================================================
    std::cout << "\n--- Test 3: Build Vector Dense ---" << std::endl;
    std::vector<float> test_embedding(dim, 0.5f);
    std::unique_ptr<fluffy::Vector> test_vector;
    fluffy::ErrorCode build_ec = fluffyInterface.build_vector_dense(test_embedding, test_vector);
    log_test("Build Vector Dense", build_ec == fluffy::ErrorCode::Success && test_vector != nullptr);
    if (build_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 4: Build Vector Sparse
    // =====================================================
    std::cout << "\n--- Test 4: Build Vector Sparse ---" << std::endl;
    std::vector<std::pair<fluffy::idx_t, float>> sparse_data = {{0, 0.5f}, {10, 0.3f}, {50, 0.8f}};
    std::unique_ptr<fluffy::Vector> sparse_vector;
    fluffy::ErrorCode sparse_ec = fluffyInterface.build_vector_sparse(sparse_data, sparse_vector, dim);
    log_test("Build Vector Sparse", sparse_ec == fluffy::ErrorCode::Success && sparse_vector != nullptr);
    if (sparse_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 5: Add Data (Single)
    // =====================================================
    std::cout << "\n--- Test 5: Add Data (Single) ---" << std::endl;
    auto vector_0 = create_test_vector(fluffyInterface, 0, dim);
    fluffy::ErrorCode add_ec = fluffyInterface.add_data("id_0", vector_0);
    log_test("Add Single Data", add_ec == fluffy::ErrorCode::Success);
    if (add_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 6: Add Data Batch
    // =====================================================
    std::cout << "\n--- Test 6: Add Data Batch ---" << std::endl;
    std::vector<std::pair<std::string, std::unique_ptr<fluffy::Vector>>> batch1;
    for (int i = 1; i <= 10; ++i) {
      std::string str_id = "id_" + std::to_string(i);
      batch1.emplace_back(str_id, create_test_vector(fluffyInterface, i, dim));
    }

    std::vector<fluffy::ErrorCode> add_ecs = fluffyInterface.add_data_batch(batch1, 4);
    bool all_adds_ok = true;
    for (const auto& ec : add_ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        all_adds_ok = false;
        break;
      }
    }
    log_test("Batch Insert (10 vectors)", all_adds_ok,
             "Added " + std::to_string(add_ecs.size()) + " vectors");
    if (!all_adds_ok) all_tests_passed = false;

    // =====================================================
    // Test 7: Get Num Unique String IDs
    // =====================================================
    std::cout << "\n--- Test 7: Get Num Unique String IDs ---" << std::endl;
    size_t unique_count = fluffyInterface.get_num_unique_string_ids();
    log_test("Unique ID Count", unique_count == 11,
             "Expected: 11, Got: " + std::to_string(unique_count));
    if (unique_count != 11) all_tests_passed = false;

    // =====================================================
    // Test 8: Get Vector Dimension
    // =====================================================
    std::cout << "\n--- Test 8: Get Vector Dimension ---" << std::endl;
    size_t vector_dim = fluffyInterface.get_vector_dim();
    log_test("Get Vector Dimension", vector_dim == dim,
             "Expected: " + std::to_string(dim) + ", Got: " + std::to_string(vector_dim));
    if (vector_dim != dim) all_tests_passed = false;

    // =====================================================
    // Test 9: Contains ID
    // =====================================================
    std::cout << "\n--- Test 9: Contains ID ---" << std::endl;
    bool contains = fluffyInterface.contains_id("id_5");
    log_test("Contains Existing ID", contains, "ID 'id_5' should exist");
    if (!contains) all_tests_passed = false;

    bool not_contains = fluffyInterface.contains_id("id_999");
    log_test("Contains Non-existent ID", !not_contains, "ID 'id_999' should not exist");
    if (not_contains) all_tests_passed = false;

    // =====================================================
    // Test 10: Flush
    // =====================================================
    std::cout << "\n--- Test 10: Flush ---" << std::endl;
    fluffy::ErrorCode flush_ec = fluffyInterface.flush();
    log_test("Flush", flush_ec == fluffy::ErrorCode::Success);
    if (flush_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 11: Search (Single Query)
    // =====================================================
    std::cout << "\n--- Test 11: Search (Single Query) ---" << std::endl;
    auto query_vector = create_test_vector(fluffyInterface, 5, dim);
    fluffy::Query query;
    query.top_k = 5;
    query.vector = query_vector.get();
    query.similarity_measure = fluffy::DistanceFunctionType::NegativeCosineSimilarity;

    fluffy::InterfaceQueryResults result;
    fluffy::ErrorCode query_ec = fluffyInterface.search(query, result, "test_index", 0.3f, 0.1f);
    log_test("Search Query", query_ec == fluffy::ErrorCode::Success);
    log_test("Query Results Count", result.results.size() > 0,
             "Results: " + std::to_string(result.results.size()));
    if (query_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 12: Search Batch
    // =====================================================
    std::cout << "\n--- Test 12: Search Batch ---" << std::endl;
    std::vector<fluffy::Query> queries;
    std::vector<fluffy::InterfaceQueryResults> results;
    std::vector<std::unique_ptr<fluffy::Vector>> query_vectors; // Keep vectors alive

    for (int i = 0; i < 3; ++i) {
      auto qv = create_test_vector(fluffyInterface, i + 20, dim);
      fluffy::Query q;
      q.top_k = 5;
      q.vector = qv.get();
      q.similarity_measure = fluffy::DistanceFunctionType::NegativeCosineSimilarity;
      queries.push_back(q);
      query_vectors.push_back(std::move(qv)); // Keep ownership
    }
    results.resize(queries.size());

    std::vector<fluffy::ErrorCode> batch_query_ecs = 
      fluffyInterface.search_batch(queries, results, "test_index", 0.3f, 0.1f, 4);

    bool all_queries_ok = true;
    for (const auto& ec : batch_query_ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        all_queries_ok = false;
        break;
      }
    }
    log_test("Batch Search", all_queries_ok,
        "Queries: " + std::to_string(batch_query_ecs.size()));
    if (!all_queries_ok) all_tests_passed = false;

    // =====================================================
    // Test 13: Get Attribute
    // =====================================================
    std::cout << "\n--- Test 13: Get Attribute ---" << std::endl;
    std::vector<std::string> attr_values;
    fluffy::ErrorCode attr_ec = fluffyInterface.get_attribute("id_5", "category", attr_values);
    log_test("Get Attribute", attr_ec == fluffy::ErrorCode::Success,
             "Values count: " + std::to_string(attr_values.size()));
    if (attr_ec == fluffy::ErrorCode::Success && !attr_values.empty()) {
      std::cout << "    Retrieved attribute value: " << attr_values[0] << std::endl;
    }
    if (attr_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 14: To Vector Dense (Retrieve)
    // =====================================================
    std::cout << "\n--- Test 14: To Vector Dense (Retrieve) ---" << std::endl;
    std::vector<std::vector<float>> retrieved_vecs;
    fluffy::ErrorCode retrieve_ec = fluffyInterface.to_vector_dense<float>("id_7", retrieved_vecs);
    log_test("Retrieve Vector Dense", retrieve_ec == fluffy::ErrorCode::Success,
             "Vectors count: " + std::to_string(retrieved_vecs.size()));
    if (retrieve_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 15: Duplicate Insert (Tagged Versions)
    // =====================================================
    std::cout << "\n--- Test 15: Duplicate Insert ---" << std::endl;
    std::vector<std::pair<std::string, std::unique_ptr<fluffy::Vector>>> batch2;
    for (int i = 5; i <= 15; ++i) {
      std::string str_id = "id_" + std::to_string(i);
      batch2.emplace_back(str_id, create_test_vector(fluffyInterface, i + 100, dim));
    }

    add_ecs = fluffyInterface.add_data_batch(batch2, 4);
    all_adds_ok = true;
    for (const auto& ec : add_ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        all_adds_ok = false;
        break;
      }
    }
    log_test("Duplicate Insert", all_adds_ok);

    unique_count = fluffyInterface.get_num_unique_string_ids();
    log_test("Unique ID Count After Duplicates", unique_count == 16,
             "Expected: 16, Got: " + std::to_string(unique_count));
    if (unique_count != 16) all_tests_passed = false;

    // =====================================================
    // Test 16: Remove Data (Single)
    // =====================================================
    std::cout << "\n--- Test 16: Remove Data (Single) ---" << std::endl;
    fluffy::ErrorCode remove_ec = fluffyInterface.remove_data("id_3");
    log_test("Remove Single Data", remove_ec == fluffy::ErrorCode::Success);
    if (remove_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 17: Remove Data (Batch)
    // =====================================================
    std::cout << "\n--- Test 17: Remove Data (Batch) ---" << std::endl;
    std::vector<std::string> remove_ids = {"id_7", "id_11", "id_15"};
    std::vector<fluffy::ErrorCode> remove_ecs = fluffyInterface.remove_data(remove_ids);

    bool all_removes_ok = true;
    for (size_t i = 0; i < remove_ecs.size(); ++i) {
      bool success = (remove_ecs[i] == fluffy::ErrorCode::Success);
      if (!success) all_removes_ok = false;
    }
    log_test("Batch Remove", all_removes_ok,
             "Removed " + std::to_string(remove_ids.size()) + " IDs");
    if (!all_removes_ok) all_tests_passed = false;

    unique_count = fluffyInterface.get_num_unique_string_ids();
    log_test("Unique Count After Removal", unique_count == 12,
             "Expected: 12, Got: " + std::to_string(unique_count));

    // =====================================================
    // Test 18: Get Num Removed Since Shrink
    // =====================================================
    std::cout << "\n--- Test 18: Get Num Removed Since Shrink ---" << std::endl;
    int num_removed = fluffyInterface.get_num_removed_since_shrink();
    log_test("Get Num Removed Since Shrink", num_removed >= 4,
             "Removed: " + std::to_string(num_removed));

    // =====================================================
    // Test 19: Shrink to Fit
    // =====================================================
    std::cout << "\n--- Test 19: Shrink to Fit ---" << std::endl;
    fluffy::ErrorCode shrink_ec = fluffyInterface.shrink_to_fit();
    log_test("Shrink to Fit", shrink_ec == fluffy::ErrorCode::Success);
    if (shrink_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    num_removed = fluffyInterface.get_num_removed_since_shrink();
    log_test("Num Removed After Shrink", num_removed == 0,
             "Expected: 0, Got: " + std::to_string(num_removed));

    // =====================================================
    // Test 20: Search After Shrink
    // =====================================================
    std::cout << "\n--- Test 20: Search After Shrink ---" << std::endl;
    fluffy::InterfaceQueryResults result2;
    query_ec = fluffyInterface.search(query, result2, "test_index", 0.3f, 0.1f);
    log_test("Search After Shrink", query_ec == fluffy::ErrorCode::Success);
    if (query_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 21: Offload
    // =====================================================
    std::cout << "\n--- Test 21: Offload ---" << std::endl;
    fluffy::ErrorCode offload_ec = fluffyInterface.offload();
    log_test("Offload", offload_ec == fluffy::ErrorCode::Success);
    if (offload_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 22: Attach SOIL Index
    // =====================================================
    std::cout << "\n--- Test 22: Attach SOIL Index ---" << std::endl;
    fluffy::ErrorCode soil_ec = fluffyInterface.attach_soil_index(
        dim, "dense", fluffy::DistanceFunctionType::NegativeCosineSimilarity,
        0.4f, "soil_index", 100000, 4);
    log_test("Attach SOIL Index", soil_ec == fluffy::ErrorCode::Success);
    if (soil_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // =====================================================
    // Test 23: Delete Index
    // =====================================================
    std::cout << "\n--- Test 23: Delete Index ---" << std::endl;
    fluffy::ErrorCode delete_ec = fluffyInterface.delete_index("test_index");
    log_test("Delete Index", delete_ec == fluffy::ErrorCode::Success);
    if (delete_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    // Try to delete non-existent index
    fluffy::ErrorCode delete_nonexist_ec = fluffyInterface.delete_index("non_existent_index");
    log_test("Delete Non-existent Index", delete_nonexist_ec != fluffy::ErrorCode::Success);

    // =====================================================
    // Test 24: Edge Cases
    // =====================================================
    std::cout << "\n--- Test 24: Edge Cases ---" << std::endl;

    // Remove non-existent ID
    fluffy::ErrorCode remove_nonexist = fluffyInterface.remove_data("id_999999");
    log_test("Remove Non-existent ID", remove_nonexist == fluffy::ErrorCode::VectorNotFound);

    // Get attribute for non-existent ID
    std::vector<std::string> attr_values2;
    fluffy::ErrorCode attr_ec2 = fluffyInterface.get_attribute("id_999999", "category", attr_values2);
    log_test("Get Attribute Non-existent ID",
             attr_ec2 == fluffy::ErrorCode::VectorNotFound || attr_ec2 != fluffy::ErrorCode::Success);

    // Query with top_k = 0
    fluffy::InterfaceQueryResults result3;
    query.top_k = 0;
    query_ec = fluffyInterface.search(query, result3, "soil_index", 0.3f, 0.1f);
    log_test("Query with top_k=0", query_ec != fluffy::ErrorCode::Success || result3.results.empty());

    // =====================================================
    // Test 25: Large Batch Operations
    // =====================================================
    std::cout << "\n--- Test 25: Large Batch Operations ---" << std::endl;
    std::vector<std::pair<std::string, std::unique_ptr<fluffy::Vector>>> large_batch;
    for (int i = 1000; i < 1100; ++i) {
      large_batch.emplace_back("id_" + std::to_string(i), create_test_vector(fluffyInterface, i, dim));
    }

    add_ecs = fluffyInterface.add_data_batch(large_batch, 8);
    all_adds_ok = true;
    for (const auto& ec : add_ecs) {
      if (ec != fluffy::ErrorCode::Success) {
        all_adds_ok = false;
        break;
      }
    }
    log_test("Large Batch Insert (100 vectors)", all_adds_ok);
    if (!all_adds_ok) all_tests_passed = false;

    // =====================================================
    // Test 26: Clear
    // =====================================================
    std::cout << "\n--- Test 26: Clear ---" << std::endl;
    size_t count_before_clear = fluffyInterface.get_num_unique_string_ids();
    log_test("Count Before Clear", count_before_clear > 0,
             "Count: " + std::to_string(count_before_clear));

    fluffy::ErrorCode clear_ec = fluffyInterface.clear();
    log_test("Clear Operation", clear_ec == fluffy::ErrorCode::Success);
    if (clear_ec != fluffy::ErrorCode::Success) all_tests_passed = false;

    size_t count_after_clear = fluffyInterface.get_num_unique_string_ids();
    log_test("Count After Clear", count_after_clear == 0,
             "Expected: 0, Got: " + std::to_string(count_after_clear));
    if (count_after_clear != 0) all_tests_passed = false;

    // =====================================================
    // Final Summary
    // =====================================================
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Overall Result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    all_tests_passed = false;
  } catch (...) {
    std::cerr << "UNKNOWN EXCEPTION occurred" << std::endl;
    all_tests_passed = false;
  }

  return all_tests_passed ? 0 : 1;
}
