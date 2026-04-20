#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <cucascade/data/data_batch.hpp>

namespace duckdb {

struct CachedCSR {
  // CSR arrays owned by this struct (allocated with cudaMalloc)
  int64_t* offsets  = nullptr;  // size: num_vertices + 1
  int64_t* indices  = nullptr;  // size: num_edges
  float*   weights  = nullptr;  // size: num_edges, nullptr if unweighted

  int64_t num_vertices = 0;
  int64_t num_edges    = 0;
  bool    is_weighted  = false;
  bool    is_built     = false;

  // Edge batches pending CSR construction; accumulated by GPUCSRConstructionOperator::execute(),
  // consumed and cleared by GPUGraphTraversalOperator::execute() on its first call.
  std::vector<std::shared_ptr<::cucascade::data_batch>> pending_batches;

  ~CachedCSR()
  {
    // Skip cudaFree if the CUDA runtime is already being torn down at process exit.
    // The OS reclaims GPU memory anyway, so leaking here is safe.
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { return; }
    cudaFree(offsets);
    cudaFree(indices);
    cudaFree(weights);
  }

  // Cache key: "EDGE_TABLE:src_col:dst_col"
  static std::string MakeKey(const std::string& edge_table,
                              const std::string& src_col,
                              const std::string& dst_col)
  {
    std::string upper_table = edge_table;
    std::transform(upper_table.begin(), upper_table.end(), upper_table.begin(), ::toupper);
    return upper_table + ":" + src_col + ":" + dst_col;
  }
};

}  // namespace duckdb
