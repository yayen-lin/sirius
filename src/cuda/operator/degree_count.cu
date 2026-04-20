#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cuda_runtime.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// degree_count_kernel
//   Processes edges in tiles of the vertex range.
//   Each block maintains a shared memory array covering only
//   [tile_start, tile_start + TILE_SIZE) vertices.
//   Edges whose src falls outside the current tile are ignored
//   and picked up in a later pass.
// ---------------------------------------------------------------------------
static constexpr int64_t TILE_SIZE = 1024;  // vertices per tile - fits in shared mem
static constexpr int BLOCK_SIZE    = 256;   // threads per block

__global__ void degree_count_kernel(const int64_t* __restrict__ src,
                                    int64_t* degree,
                                    int64_t num_edges,
                                    int64_t tile_start)
{
  __shared__ unsigned long long shared_degree[TILE_SIZE];

  // Initialize shared tile to zero
  for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
    shared_degree[i] = 0;
  }
  __syncthreads();

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_edges) {
    int64_t v     = src[idx];
    int64_t local = v - tile_start;
    if (local >= 0 && local < TILE_SIZE) {
      atomicAdd(&shared_degree[local], static_cast<unsigned long long>(1));
    }
  }
  __syncthreads();

  // Flush shared tile back to global degree array
  for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
    if (shared_degree[i] > 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&degree[tile_start + i]), shared_degree[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// LaunchDegreeCountKernel
//   Iterates over vertex tiles. For each tile, launches one kernel pass
//   over all edges. Edges outside the current tile are skipped by the kernel.
// ---------------------------------------------------------------------------
void LaunchDegreeCountKernel(const int64_t* src,
                             int64_t* degree,
                             int64_t num_edges,
                             int64_t num_vertices)
{
  // TODO: sort src/dst by src vertex before tiled degree counting
  // to eliminate redundant edge scans per tile. Use CUB DeviceRadixSort.
  // Currently O(num_tiles * num_edges); after sorting O(num_edges).
  if (num_edges == 0) return;

  cudaMemset(degree, 0, num_vertices * sizeof(int64_t));

  int64_t num_blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int64_t num_tiles  = (num_vertices + TILE_SIZE - 1) / TILE_SIZE;

  for (int64_t tile = 0; tile < num_tiles; tile++) {
    int64_t tile_start = tile * TILE_SIZE;
    degree_count_kernel<<<num_blocks, BLOCK_SIZE>>>(src, degree, num_edges, tile_start);
  }
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// LaunchFindMaxKernel
//   Returns the maximum value in a device int64_t array.
//   Uses thrust on the CUDA execution policy; safe to call from .cu files.
//   Allocates via thrust's CUB backend (cudaMalloc), not through RMM.
// ---------------------------------------------------------------------------
int64_t LaunchFindMaxKernel(const int64_t* data, int64_t n)
{
  if (n == 0) { return -1; }
  auto ptr = thrust::device_ptr<const int64_t>(data);
  auto it  = thrust::max_element(thrust::cuda::par, ptr, ptr + n);
  cudaDeviceSynchronize();
  return *it;  // thrust::device_ptr dereference copies one value to host
}

}  // namespace duckdb