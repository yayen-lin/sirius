#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// LaunchEdgeSortKernel
//   sorts edge list by source vertex, keeping dst paired, using cub
//
//   src_in/dst_in: input edge arrays
//   src_out/dst_out: output sorted arrays
//   num_edges: number of edges
//   stream: CUDA stream for async execution
// ---------------------------------------------------------------------------
void LaunchEdgeSortKernel(const int64_t* src_in,
                          const int64_t* dst_in,
                          int64_t* src_out,
                          int64_t* dst_out,
                          int64_t num_edges,
                          rmm::cuda_stream_view stream)
{
  if (num_edges == 0) return;

  // determine temporary storage size for CUB radix sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  src_in,
                                  src_out,
                                  dst_in,
                                  dst_out,
                                  num_edges,
                                  0,
                                  sizeof(int64_t) * 8,
                                  stream.value());

  // temp storage
  rmm::device_uvector<uint8_t> temp_storage(temp_storage_bytes, stream);

  // sort edges by source vertex and keep dst paired
  d_temp_storage = temp_storage.data();
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  src_in,
                                  src_out,
                                  dst_in,
                                  dst_out,
                                  num_edges,
                                  0,
                                  sizeof(int64_t) * 8,
                                  stream.value());

  // synchronize to ensure sort completes before downstream kernels read sorted data
  cudaStreamSynchronize(stream.value());
}

}  // namespace duckdb
