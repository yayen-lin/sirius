#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace duckdb {

// ---------------------------------------------------------------------------
// fill_int64_kernel - fills device array with a constant value
// ---------------------------------------------------------------------------
__global__ void fill_int64_kernel(int64_t* __restrict__ data, int64_t value, size_t n)
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) { data[idx] = value; }
}

// ---------------------------------------------------------------------------
// LaunchFillKernel - GPU-native constant fill (stream-ordered)
// ---------------------------------------------------------------------------
void LaunchFillKernel(int64_t* data, int64_t value, size_t n, rmm::cuda_stream_view stream)
{
  if (n == 0) return;

  constexpr int BLOCK_SIZE = 256;
  size_t num_blocks        = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  fill_int64_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data, value, n);
}

}  // namespace duckdb
