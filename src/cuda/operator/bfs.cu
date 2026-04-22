#include "utils.hpp"

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace duckdb {

// ---------------------------------------------------------------------------
// BFSResult - RAII container for BFS output
// ---------------------------------------------------------------------------
struct BFSResult {
  rmm::device_uvector<int64_t> node_ids;
  rmm::device_uvector<int64_t> distances;
  rmm::device_uvector<int64_t> predecessors;

  BFSResult(rmm::device_uvector<int64_t>&& nodes,
           rmm::device_uvector<int64_t>&& dists,
           rmm::device_uvector<int64_t>&& preds)
    : node_ids(std::move(nodes)), distances(std::move(dists)), predecessors(std::move(preds)) {}
};

static constexpr int64_t UNVISITED = -1;

// ---------------------------------------------------------------------------
// bfs_init_kernel
//   Marks source vertices as visited (distance=0, predecessor=self)
//   and adds them to the initial frontier.
// ---------------------------------------------------------------------------
__global__ void bfs_init_kernel(const int64_t* __restrict__ source_ids,
                                int64_t* visited,
                                int64_t* predecessor,
                                int64_t* frontier,
                                int64_t* frontier_size,
                                int64_t num_sources)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_sources) return;

  int64_t src = source_ids[idx];

  // Only mark if not already visited (handles duplicate source IDs)
  int64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(&visited[src]),
                          static_cast<unsigned long long>(UNVISITED),
                          static_cast<unsigned long long>(0));
  if (old == UNVISITED) {
    predecessor[src] = src;
    int64_t pos      = atomicAdd(reinterpret_cast<unsigned long long*>(frontier_size),
                            static_cast<unsigned long long>(1));
    frontier[pos]    = src;
  }
}

// ---------------------------------------------------------------------------
// bfs_expand_kernel
//   One thread per vertex in current frontier.
//   For each neighbor, attempts to mark it visited via atomicCAS.
//   If successful, writes predecessor and adds to next frontier.
// ---------------------------------------------------------------------------
__global__ void bfs_expand_kernel(const int64_t* __restrict__ csr_offsets,
                                  const int64_t* __restrict__ csr_indices,
                                  const int64_t* __restrict__ current_frontier,
                                  int64_t* next_frontier,
                                  int64_t* next_frontier_size,
                                  int64_t* visited,
                                  int64_t* predecessor,
                                  int64_t current_frontier_size,
                                  int64_t current_distance)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= current_frontier_size) return;

  int64_t v         = current_frontier[idx];
  int64_t nbr_start = csr_offsets[v];
  int64_t nbr_end   = csr_offsets[v + 1];

  for (int64_t n = nbr_start; n < nbr_end; n++) {
    int64_t neighbor = csr_indices[n];

    // Attempt to claim this neighbor - only first thread to reach it wins
    int64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(&visited[neighbor]),
                            static_cast<unsigned long long>(UNVISITED),
                            static_cast<unsigned long long>(current_distance + 1));

    if (old == UNVISITED) {
      predecessor[neighbor] = v;
      int64_t pos           = atomicAdd(reinterpret_cast<unsigned long long*>(next_frontier_size),
                              static_cast<unsigned long long>(1));
      next_frontier[pos]    = neighbor;
    }
  }
}

// ---------------------------------------------------------------------------
// LaunchBFSKernel
//   Initialises BFS from source vertices, then expands level by level
//   until the frontier is empty.
//
//   Outputs: BFSResult containing node_ids, distances, and predecessors
//   (all reached vertices including sources)
//
//   Now uses RMM for memory management and returns device_uvectors.
// ---------------------------------------------------------------------------
BFSResult LaunchBFSKernel(const int64_t* csr_offsets,
                          const int64_t* csr_indices,
                          const int64_t* source_ids,
                          int64_t num_sources,
                          int64_t num_vertices,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  if (num_sources == 0 || num_vertices == 0) {
    return BFSResult(rmm::device_uvector<int64_t>(0, stream, mr),
                    rmm::device_uvector<int64_t>(0, stream, mr),
                    rmm::device_uvector<int64_t>(0, stream, mr));
  }

  constexpr int BLOCK_SIZE = 256;

  // --- Allocate BFS state arrays with RMM ---
  rmm::device_uvector<int64_t> visited(num_vertices, stream, mr);
  rmm::device_uvector<int64_t> predecessor(num_vertices, stream, mr);

  // Initialise visited to UNVISITED (-1): all bytes 0xFF → int64_t = -1 (two's complement)
  cudaMemsetAsync(visited.data(), 0xFF, num_vertices * sizeof(int64_t), stream.value());
  cudaMemsetAsync(predecessor.data(), 0xFF, num_vertices * sizeof(int64_t), stream.value());

  // Frontier ping-pong buffers - worst case all vertices in frontier
  rmm::device_uvector<int64_t> frontier_a(num_vertices, stream, mr);
  rmm::device_uvector<int64_t> frontier_b(num_vertices, stream, mr);
  rmm::device_uvector<int64_t> frontier_size(1, stream, mr);

  // --- Initialise frontier with source vertices ---
  cudaMemsetAsync(frontier_size.data(), 0, sizeof(int64_t), stream.value());

  int64_t init_blocks = (num_sources + BLOCK_SIZE - 1) / BLOCK_SIZE;
  bfs_init_kernel<<<init_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    source_ids, visited.data(), predecessor.data(), frontier_a.data(), frontier_size.data(), num_sources);

  int64_t h_frontier_size = 0;
  cudaMemcpyAsync(&h_frontier_size, frontier_size.data(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  // --- BFS level-by-level expansion ---
  int64_t* current_frontier = frontier_a.data();
  int64_t* next_frontier    = frontier_b.data();
  int64_t current_distance  = 0;

  while (h_frontier_size > 0) {
    // Reset next frontier size
    cudaMemsetAsync(frontier_size.data(), 0, sizeof(int64_t), stream.value());

    int64_t expand_blocks = (h_frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bfs_expand_kernel<<<expand_blocks, BLOCK_SIZE, 0, stream.value()>>>(
      csr_offsets,
      csr_indices,
      current_frontier,
      next_frontier,
      frontier_size.data(),
      visited.data(),
      predecessor.data(),
      h_frontier_size,
      current_distance);

    // Read next frontier size back to CPU to control the loop
    cudaMemcpyAsync(&h_frontier_size, frontier_size.data(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();

    // Swap frontiers
    std::swap(current_frontier, next_frontier);
    current_distance++;
  }

  // --- Collect results - all vertices where visited != UNVISITED ---
  // CPU-side collection for prototype correctness
  // TODO: replace with GPU compaction kernel (stream compaction)
  std::vector<int64_t> h_visited(num_vertices);
  std::vector<int64_t> h_predecessor(num_vertices);
  cudaMemcpyAsync(h_visited.data(), visited.data(), num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value());
  cudaMemcpyAsync(h_predecessor.data(), predecessor.data(), num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  std::vector<int64_t> h_node_ids, h_distances, h_predecessors;
  for (int64_t v = 0; v < num_vertices; v++) {
    if (h_visited[v] != UNVISITED) {
      h_node_ids.push_back(v);
      h_distances.push_back(h_visited[v]);
      h_predecessors.push_back(h_predecessor[v]);
    }
  }

  int64_t out_count = static_cast<int64_t>(h_node_ids.size());

  // Copy results back to GPU with RMM
  rmm::device_uvector<int64_t> out_node_ids(out_count, stream, mr);
  rmm::device_uvector<int64_t> out_distances(out_count, stream, mr);
  rmm::device_uvector<int64_t> out_predecessors(out_count, stream, mr);

  cudaMemcpyAsync(out_node_ids.data(), h_node_ids.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream.value());
  cudaMemcpyAsync(out_distances.data(), h_distances.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream.value());
  cudaMemcpyAsync(out_predecessors.data(), h_predecessors.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice, stream.value());

  // visited, predecessor, frontiers freed automatically

  return BFSResult(std::move(out_node_ids), std::move(out_distances), std::move(out_predecessors));
}

}  // namespace duckdb