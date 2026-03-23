#include "gpu_buffer_manager.hpp"

#include <cuda_runtime.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// scatter_kernel
//   One thread per edge.
//   write_cursors is a mutable copy of offsets - each thread atomically
//   claims the next available slot for its source vertex.
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(const int64_t* __restrict__ src,
                               const int64_t* __restrict__ dst,
                               int64_t* write_cursors,
                               int64_t* indices,
                               int64_t num_edges)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_edges) return;

  int64_t src_vertex = src[idx];
  // Atomically claim the next slot in this vertex's adjacency list
  int64_t pos  = atomicAdd(reinterpret_cast<unsigned long long*>(&write_cursors[src_vertex]),
                          static_cast<unsigned long long>(1));
  indices[pos] = dst[idx];
}

// ---------------------------------------------------------------------------
// LaunchScatterKernel
//   Allocates a temporary write_cursors array (copy of offsets),
//   launches the kernel, then frees write_cursors.
//   offsets is left intact for the traversal operator.
// ---------------------------------------------------------------------------
void LaunchScatterKernel(const int64_t* src,
                         const int64_t* dst,
                         const int64_t* offsets,
                         int64_t* indices,
                         int64_t num_edges,
                         int64_t num_vertices)
{
  if (num_edges == 0) return;

  auto& mgr = GPUBufferManager::GetInstance();

  // Allocate temporary write cursors - copy of offsets, consumed during scatter
  auto* write_cursors = mgr.customCudaMalloc<int64_t>(num_vertices + 1, 0, false);
  cudaMemcpy(
    write_cursors, offsets, (num_vertices + 1) * sizeof(int64_t), cudaMemcpyDeviceToDevice);

  constexpr int BLOCK_SIZE = 256;
  int64_t num_blocks       = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

  scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(src, dst, write_cursors, indices, num_edges);
  cudaDeviceSynchronize();

  mgr.customCudaFree(reinterpret_cast<uint8_t*>(write_cursors), 0);
}

}  // namespace duckdb