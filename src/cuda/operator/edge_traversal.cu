#include "gpu_buffer_manager.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// edge_traversal_kernel
//   One thread per source vertex.
//   For each source, reads its neighbor range from offsets and writes
//   all neighbors into out_node_ids and out_predecessor_ids.
//   Uses a pre-computed per-source write offset (src_offsets_out) so
//   threads can scatter without contention.
// ---------------------------------------------------------------------------
__global__ void edge_traversal_kernel(const int64_t* __restrict__ csr_offsets,
                                      const int64_t* __restrict__ csr_indices,
                                      const int64_t* __restrict__ source_ids,
                                      const int64_t* __restrict__ src_offsets_out,
                                      int64_t* out_node_ids,
                                      int64_t* out_predecessor_ids,
                                      int64_t num_sources)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_sources) return;

  int64_t src        = source_ids[idx];
  int64_t nbr_start  = csr_offsets[src];
  int64_t nbr_end    = csr_offsets[src + 1];
  int64_t write_base = src_offsets_out[idx];

  for (int64_t n = nbr_start; n < nbr_end; n++) {
    int64_t pos              = write_base + (n - nbr_start);
    out_node_ids[pos]        = csr_indices[n];
    out_predecessor_ids[pos] = src;
  }
}

// ---------------------------------------------------------------------------
// compute_src_degrees_kernel
//   One thread per source vertex.
//   Writes degree of each source into src_degrees[idx].
// ---------------------------------------------------------------------------
__global__ void compute_src_degrees_kernel(const int64_t* __restrict__ csr_offsets,
                                           const int64_t* __restrict__ source_ids,
                                           int64_t* src_degrees,
                                           int64_t num_sources)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_sources) return;

  int64_t src      = source_ids[idx];
  src_degrees[idx] = csr_offsets[src + 1] - csr_offsets[src];
}

// ---------------------------------------------------------------------------
// LaunchEdgeTraversalKernel
//   1. Compute degree of each source vertex
//   2. CPU exclusive scan of degrees → per-source write offsets
//   3. Allocate output arrays (total_neighbors entries)
//   4. Launch edge_traversal_kernel to scatter neighbors
// ---------------------------------------------------------------------------
void LaunchEdgeTraversalKernel(const int64_t* csr_offsets,
                               const int64_t* csr_indices,
                               const int64_t* source_ids,
                               int64_t num_sources,
                               int64_t*& out_node_ids,
                               int64_t*& out_predecessor_ids,
                               int64_t& out_count)
{
  if (num_sources == 0) {
    out_node_ids        = nullptr;
    out_predecessor_ids = nullptr;
    out_count           = 0;
    return;
  }

  auto& mgr = GPUBufferManager::GetInstance();

  constexpr int BLOCK_SIZE = 256;
  int64_t num_blocks       = (num_sources + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // --- Step 1: compute per-source degrees ---
  int64_t* src_degrees = mgr.customCudaMalloc<int64_t>(num_sources, 0, false);
  compute_src_degrees_kernel<<<num_blocks, BLOCK_SIZE>>>(
    csr_offsets, source_ids, src_degrees, num_sources);
  cudaDeviceSynchronize();

  // --- Step 2: CPU exclusive scan of degrees → write offsets ---
  // num_sources is small (literal ID list), so CPU scan is fine here
  std::vector<int64_t> h_degrees(num_sources);
  cudaMemcpy(h_degrees.data(), src_degrees, num_sources * sizeof(int64_t), cudaMemcpyDeviceToHost);

  std::vector<int64_t> h_src_offsets_out(num_sources);
  h_src_offsets_out[0] = 0;
  for (int64_t i = 1; i < num_sources; i++) {
    h_src_offsets_out[i] = h_src_offsets_out[i - 1] + h_degrees[i - 1];
  }
  int64_t total_neighbors = h_src_offsets_out[num_sources - 1] + h_degrees[num_sources - 1];

  // --- Step 3: allocate output arrays ---
  int64_t* d_src_offsets_out = mgr.customCudaMalloc<int64_t>(num_sources, 0, false);
  cudaMemcpy(d_src_offsets_out,
             h_src_offsets_out.data(),
             num_sources * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  out_node_ids        = mgr.customCudaMalloc<int64_t>(total_neighbors, 0, false);
  out_predecessor_ids = mgr.customCudaMalloc<int64_t>(total_neighbors, 0, false);

  // --- Step 4: scatter neighbors ---
  edge_traversal_kernel<<<num_blocks, BLOCK_SIZE>>>(csr_offsets,
                                                    csr_indices,
                                                    source_ids,
                                                    d_src_offsets_out,
                                                    out_node_ids,
                                                    out_predecessor_ids,
                                                    num_sources);
  cudaDeviceSynchronize();

  mgr.customCudaFree(reinterpret_cast<uint8_t*>(src_degrees), 0);
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(d_src_offsets_out), 0);

  out_count = total_neighbors;
}

}  // namespace duckdb