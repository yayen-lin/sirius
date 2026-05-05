#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// compute_vertex_starts_kernel
//   For sorted edges, marks where each vertex's edges begin.
//   vertex_starts[v] = index of first edge with src=v in sorted array.
// ---------------------------------------------------------------------------
__global__ void compute_vertex_starts_kernel(const int64_t* __restrict__ src,
                                             int64_t* vertex_starts,
                                             int64_t num_edges,
                                             int64_t num_vertices)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_edges) return;

  int64_t src_vertex = src[idx];

  // First edge overall, or first edge of a new vertex
  bool is_first = (idx == 0) || (src[idx - 1] != src_vertex);

  if (is_first) {
    vertex_starts[src_vertex] = idx;
  }
}

// ---------------------------------------------------------------------------
// scatter_kernel
//   computes position using vertex_starts with sorted edges
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(const int64_t* __restrict__ src,
                                          const int64_t* __restrict__ dst,
                                          const int64_t* __restrict__ offsets,
                                          const int64_t* __restrict__ vertex_starts,
                                          int64_t* indices,
                                          int64_t num_edges)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_edges) return;

  int64_t src_vertex = src[idx];

  // Local offset within this vertex's edge list
  int64_t local_offset = idx - vertex_starts[src_vertex];

  // Write position in CSR indices array
  int64_t write_pos = offsets[src_vertex] + local_offset;

  indices[write_pos] = dst[idx];
}

// ---------------------------------------------------------------------------
// LaunchScatterKernel
//   Two-pass:
//   1. Compute vertex_starts array
//   2. Scatter using vertex_starts to compute positions
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

  constexpr int BLOCK_SIZE = 256;
  int64_t num_blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // pass 1: compute vertex_starts array
  rmm::device_uvector<int64_t> vertex_starts(num_vertices, stream, mr);

  // initialize to -1 for vertices with no edges
  cudaMemsetAsync(vertex_starts.data(), -1, num_vertices * sizeof(int64_t), stream.value());

  compute_vertex_starts_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    src, vertex_starts.data(), num_edges, num_vertices);

  // pass 2: scatter
  scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    src, dst, offsets, vertex_starts.data(), indices, num_edges);
}

}  // namespace duckdb
