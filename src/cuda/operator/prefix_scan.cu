#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace duckdb {

static constexpr int SCAN_BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// reduce_kernel  (pass 1)
//   Each block computes the sum of its assigned segment of `degree`
//   and writes it to `block_sums[blockIdx.x]`.
// ---------------------------------------------------------------------------
__global__ void reduce_kernel(const int64_t* __restrict__ degree,
                              int64_t* block_sums,
                              int64_t num_vertices)
{
  __shared__ int64_t shared[SCAN_BLOCK_SIZE];

  int64_t idx         = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = (idx < num_vertices) ? degree[idx] : 0;
  __syncthreads();

  // Reduction within block (tree reduction)
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) { shared[threadIdx.x] += shared[threadIdx.x + stride]; }
    __syncthreads();
  }

  if (threadIdx.x == 0) { block_sums[blockIdx.x] = shared[0]; }
}

// ---------------------------------------------------------------------------
// down_sweep_kernel  (pass 2)
//   Uses block_sums (prefix-scanned on CPU between passes) as the
//   base offset for each block, then performs an exclusive scan
//   within the block using shared memory.
//   Writes results to offsets[1..num_vertices], with offsets[0] = 0
//   enforced by the launcher.
// ---------------------------------------------------------------------------
__global__ void down_sweep_kernel(const int64_t* __restrict__ degree,
                                  const int64_t* __restrict__ block_sums_scanned,
                                  int64_t* offsets,
                                  int64_t num_vertices)
{
  __shared__ int64_t shared[SCAN_BLOCK_SIZE];

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  shared[threadIdx.x] = (idx < num_vertices) ? degree[idx] : 0;
  __syncthreads();

  // Inclusive prefix scan within block (Hillis-Steele up-sweep)
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    int64_t val = 0;
    if (threadIdx.x >= stride) { val = shared[threadIdx.x - stride]; }
    __syncthreads();
    if (threadIdx.x >= stride) { shared[threadIdx.x] += val; }
    __syncthreads();
  }

  // offsets[idx+1] = inclusive prefix sum up to vertex idx = sum(degree[0..idx])
  // offsets[0] = 0 is set explicitly by the launcher before this kernel runs.
  int64_t base = block_sums_scanned[blockIdx.x];
  if (idx < num_vertices) { offsets[idx + 1] = shared[threadIdx.x] + base; }
}

// ---------------------------------------------------------------------------
// LaunchPrefixScanKernel
//   Pass 1: reduce_kernel computes per-block sums
//   Between passes: CPU exclusive scan of block_sums (small array)
//   Pass 2: down_sweep_kernel uses scanned block sums as base offsets
//
//   Now uses RMM for memory management and stream-ordered execution.
// ---------------------------------------------------------------------------
void LaunchPrefixScanKernel(const int64_t* degree,
                            int64_t* offsets,
                            int64_t num_vertices,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (num_vertices == 0) return;

  int64_t num_blocks = (num_vertices + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

  // Allocate block_sums with RMM — integrates with Sirius memory management
  rmm::device_uvector<int64_t> block_sums(num_blocks, stream, mr);

  // --- Pass 1: per-block reduction ---
  reduce_kernel<<<num_blocks, SCAN_BLOCK_SIZE, 0, stream.value()>>>(
    degree, block_sums.data(), num_vertices);

  // --- Between passes: CPU exclusive scan of block_sums ---
  // block_sums is small (num_blocks entries), so CPU scan is fine here.
  // This is the inter-block prefix that gives each block its base offset.
  std::vector<int64_t> h_block_sums(num_blocks);
  cudaMemcpyAsync(h_block_sums.data(),
                  block_sums.data(),
                  num_blocks * sizeof(int64_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();  // Wait for D2H copy to complete before CPU scan

  std::vector<int64_t> h_block_sums_scanned(num_blocks);
  h_block_sums_scanned[0] = 0;
  for (int64_t i = 1; i < num_blocks; i++) {
    h_block_sums_scanned[i] = h_block_sums_scanned[i - 1] + h_block_sums[i - 1];
  }

  cudaMemcpyAsync(block_sums.data(),
                  h_block_sums_scanned.data(),
                  num_blocks * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // --- Pass 2: down-sweep within each block using base offsets ---
  // offsets[0] = 0 enforced here before kernel writes offsets[1..num_vertices]
  cudaMemsetAsync(offsets, 0, sizeof(int64_t), stream.value());

  down_sweep_kernel<<<num_blocks, SCAN_BLOCK_SIZE, 0, stream.value()>>>(
    degree, block_sums.data(), offsets, num_vertices);

  // block_sums freed automatically when device_uvector goes out of scope
}


// void LaunchPrefixScanKernel(const int64_t* degree, int64_t* offsets, int64_t num_vertices) {
//    TODO: cub comparison, cub::DeviceScan::ExclusiveSum(degree, offsets+1, num_vertices)
// }

}  // namespace duckdb