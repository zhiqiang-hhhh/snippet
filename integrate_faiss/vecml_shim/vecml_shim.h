// C shim for VecML (Fluffy) to avoid C++ ABI conflicts.
#pragma once

#include <cstddef>

extern "C" {

// Opaque context for VecML instance
typedef void* vecml_ctx_t;

// Create and return a VecML context. Returns nullptr on error.
vecml_ctx_t vecml_create(const char* base_path, const char* license_path);

// Add data: `data` is n * dim floats, `ids` is array of n int64 ids (can be nullptr to use 0..n-1).
// Returns 0 on success, non-zero on error.
int vecml_add_data(vecml_ctx_t ctx, const float* data, int n, int dim, const long* ids);

// Search: queries is nq * dim floats; out_ids must be preallocated with size nq * k
// and will be filled with long ids (or -1 for missing). Returns 0 on success.
int vecml_search(vecml_ctx_t ctx, const float* queries, int nq, int dim, int k, long* out_ids);

// Destroy context
void vecml_destroy(vecml_ctx_t ctx);

// Return disk usage (megabytes) under the VecML base path used to create the context.
// Returns a non-negative double on success, or -1.0 on error.
double vecml_get_disk_mb(vecml_ctx_t ctx);

} // extern C
