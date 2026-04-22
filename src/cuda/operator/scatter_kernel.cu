#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

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
//
//   Now uses RMM for memory management and stream-ordered execution.
// ---------------------------------------------------------------------------
void LaunchScatterKernel(const int64_t* src,
                         const int64_t* dst,
                         const int64_t* offsets,
                         int64_t* indices,
                         int64_t num_edges,
                         int64_t num_vertices,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  if (num_edges == 0) return;

  // Allocate temporary write cursors with RMM — integrates with Sirius memory management
  rmm::device_uvector<int64_t> write_cursors(num_vertices + 1, stream, mr);
  cudaMemcpyAsync(write_cursors.data(),
                  offsets,
                  (num_vertices + 1) * sizeof(int64_t),
                  cudaMemcpyDeviceToDevice,
                  stream.value());

  constexpr int BLOCK_SIZE = 256;
  int64_t num_blocks       = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

  scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    src, dst, write_cursors.data(), indices, num_edges);

  // write_cursors freed automatically when device_uvector goes out of scope
}

}  // namespace duckdb