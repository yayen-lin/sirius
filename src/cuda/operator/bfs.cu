#include "gpu_buffer_manager.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>

namespace duckdb {

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
//   Outputs:
//     out_node_ids       - all reached vertices (excluding sources)
//     out_distances      - distance from nearest source
//     out_predecessors   - predecessor on shortest path
//     out_count          - number of reached vertices
// ---------------------------------------------------------------------------
void LaunchBFSKernel(const int64_t* csr_offsets,
                     const int64_t* csr_indices,
                     const int64_t* source_ids,
                     int64_t num_sources,
                     int64_t num_vertices,
                     int64_t*& out_node_ids,
                     int64_t*& out_distances,
                     int64_t*& out_predecessors,
                     int64_t& out_count)
{
  if (num_sources == 0 || num_vertices == 0) {
    out_node_ids     = nullptr;
    out_distances    = nullptr;
    out_predecessors = nullptr;
    out_count        = 0;
    return;
  }

  auto& mgr = GPUBufferManager::GetInstance();

  constexpr int BLOCK_SIZE = 256;

  // --- Allocate BFS state arrays ---
  int64_t* visited     = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);
  int64_t* predecessor = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);

  // Initialise visited to UNVISITED (-1)
  // cudaMemset sets bytes not values, so we use a fill via cudaMemset(-1)
  // which sets all bytes to 0xFF - for int64_t this gives -1 (two's complement)
  cudaMemset(visited, 0xFF, num_vertices * sizeof(int64_t));
  cudaMemset(predecessor, 0xFF, num_vertices * sizeof(int64_t));

  // Frontier ping-pong buffers - worst case all vertices in frontier
  int64_t* frontier_a    = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);
  int64_t* frontier_b    = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);
  int64_t* frontier_size = mgr.customCudaMalloc<int64_t>(1, 0, /*caching=*/false);

  // --- Initialise frontier with source vertices ---
  cudaMemset(frontier_size, 0, sizeof(int64_t));

  int64_t init_blocks = (num_sources + BLOCK_SIZE - 1) / BLOCK_SIZE;
  bfs_init_kernel<<<init_blocks, BLOCK_SIZE>>>(
    source_ids, visited, predecessor, frontier_a, frontier_size, num_sources);
  cudaDeviceSynchronize();

  int64_t h_frontier_size = 0;
  cudaMemcpy(&h_frontier_size, frontier_size, sizeof(int64_t), cudaMemcpyDeviceToHost);

  // --- BFS level-by-level expansion ---
  int64_t* current_frontier = frontier_a;
  int64_t* next_frontier    = frontier_b;
  int64_t current_distance  = 0;

  while (h_frontier_size > 0) {
    // Reset next frontier size
    cudaMemset(frontier_size, 0, sizeof(int64_t));

    int64_t expand_blocks = (h_frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bfs_expand_kernel<<<expand_blocks, BLOCK_SIZE>>>(csr_offsets,
                                                     csr_indices,
                                                     current_frontier,
                                                     next_frontier,
                                                     frontier_size,
                                                     visited,
                                                     predecessor,
                                                     h_frontier_size,
                                                     current_distance);
    cudaDeviceSynchronize();

    // Read next frontier size back to CPU to control the loop
    cudaMemcpy(&h_frontier_size, frontier_size, sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Swap frontiers
    std::swap(current_frontier, next_frontier);
    current_distance++;
  }

  // --- Collect results - all vertices where visited != UNVISITED ---
  // CPU-side collection for prototype correctness
  // TODO: replace with GPU compaction kernel (stream compaction)
  std::vector<int64_t> h_visited(num_vertices);
  std::vector<int64_t> h_predecessor(num_vertices);
  cudaMemcpy(h_visited.data(), visited, num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(
    h_predecessor.data(), predecessor, num_vertices * sizeof(int64_t), cudaMemcpyDeviceToHost);

  std::vector<int64_t> h_node_ids, h_distances, h_predecessors;
  for (int64_t v = 0; v < num_vertices; v++) {
    if (h_visited[v] != UNVISITED) {
      h_node_ids.push_back(v);
      h_distances.push_back(h_visited[v]);
      h_predecessors.push_back(h_predecessor[v]);
    }
  }

  out_count = static_cast<int64_t>(h_node_ids.size());

  // Copy results back to GPU for output_relation
  out_node_ids     = mgr.customCudaMalloc<int64_t>(out_count, 0, /*caching=*/false);
  out_distances    = mgr.customCudaMalloc<int64_t>(out_count, 0, /*caching=*/false);
  out_predecessors = mgr.customCudaMalloc<int64_t>(out_count, 0, /*caching=*/false);

  cudaMemcpy(out_node_ids, h_node_ids.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    out_distances, h_distances.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    out_predecessors, h_predecessors.data(), out_count * sizeof(int64_t), cudaMemcpyHostToDevice);

  // --- Free BFS state ---
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(visited), 0);
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(predecessor), 0);
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(frontier_a), 0);
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(frontier_b), 0);
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(frontier_size), 0);
}

}  // namespace duckdb