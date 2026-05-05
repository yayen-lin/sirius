#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// degree_count_sorted_kernel
//   Counts vertex degrees from a sorted src array in a single pass.
//   Each thread checks if it's at a boundary and writes the count directly
// ---------------------------------------------------------------------------
static constexpr int BLOCK_SIZE = 256;

__global__ void degree_count_sorted_kernel(const int64_t* __restrict__ src,
                                           int64_t* degree,
                                           int64_t num_edges)
{
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_edges) return;

  int64_t current_vertex = src[idx];

  // boundary check, last occurrence of current_vertex
  bool is_last = (idx == num_edges - 1) || (src[idx + 1] != current_vertex);

  if (is_last) {
    // find the first occurrence by scanning backwards
    int64_t first = idx;
    while (first > 0 && src[first - 1] == current_vertex) {
      first--;
    }
    int64_t count          = idx - first + 1;
    degree[current_vertex] = count;
  }
}

// ---------------------------------------------------------------------------
// LaunchDegreeCountKernel
//   assumes src array is already SORTED by source vertex.
//   counts degrees in a single pass.
// ---------------------------------------------------------------------------
void LaunchDegreeCountKernel(const int64_t* src,
                             int64_t* degree,
                             int64_t num_edges,
                             int64_t num_vertices)
{
  if (num_edges == 0) return;

  cudaMemset(degree, 0, num_vertices * sizeof(int64_t));

  int64_t num_blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
  degree_count_sorted_kernel<<<num_blocks, BLOCK_SIZE>>>(src, degree, num_edges);

  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// LaunchFindMaxKernel
//   Returns the maximum value in a device int64_t array with thrust
//   Allocates via thrust's CUB backend (cudaMalloc), not through RMM
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