#include "graph/GPUCSRConstructionOperator.hpp"

#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace sirius::op {

using namespace duckdb;

GPUCSRConstructionOperator::GPUCSRConstructionOperator(vector<LogicalType> types,
                                                       const ParsedGraphQuery& parsed,
                                                       idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CSR_CONSTRUCTION, std::move(types), estimated_cardinality),
    parsed(parsed)
{
}

std::string GPUCSRConstructionOperator::params_to_string() const
{
  return " [edge_table=" + parsed.edge_table +
         ", src=" + (parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col) +
         ", dst=" + (parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col) + "]";
}

std::unique_ptr<operator_data> GPUCSRConstructionOperator::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  // pass through empty data
  return std::make_unique<operator_data>();
}

// receives edge data and builds the CSR
void GPUCSRConstructionOperator::sink(const operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto start = high_resolution_clock::now();

  auto& mgr        = GPUBufferManager::GetInstance();
  const string key = CachedCSR::MakeKey(parsed.edge_table,
                                        parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col,
                                        parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col);

  // cache hit - reuse existing CSR, skip construction entirely
  auto it = mgr.csr_cache.find(key);
  if (it != mgr.csr_cache.end()) {
    csr = it->second;
    SIRIUS_LOG_INFO("CSR cache hit for key '{}', skipping construction", key);
    return;
  }

  // cache miss - read edge columns directly from buffer manager
  SIRIUS_LOG_INFO("CSR cache miss for key '{}', building CSR from cached table", key);

  string up_table = parsed.edge_table;
  std::ranges::transform(up_table.begin(), up_table.end(), up_table.begin(), ::toupper);

  string src_col = parsed.edge_src_col.empty() ? "SRC" : parsed.edge_src_col;
  string dst_col = parsed.edge_dst_col.empty() ? "DST" : parsed.edge_dst_col;
  std::ranges::transform(src_col.begin(), src_col.end(), src_col.begin(), ::toupper);
  std::ranges::transform(dst_col.begin(), dst_col.end(), dst_col.begin(), ::toupper);

  if (!mgr.checkIfColumnCached(up_table, src_col) || !mgr.checkIfColumnCached(up_table, dst_col)) {
    throw InvalidInputException(
      "Edge table '{}' with columns '{}'/'{}' not cached. "
      "Call gpu_execution('SELECT * FROM {}') first.",
      parsed.edge_table,
      src_col,
      dst_col,
      parsed.edge_table);
  }

  auto& table   = mgr.tables[up_table];
  auto src_it   = std::find(table->column_names.begin(), table->column_names.end(), src_col);
  auto dst_it   = std::find(table->column_names.begin(), table->column_names.end(), dst_col);
  idx_t src_idx = std::distance(table->column_names.begin(), src_it);
  idx_t dst_idx = std::distance(table->column_names.begin(), dst_it);

  // convert cached GPU columns to data_batch format for BuildCSR
  // right now create a simple wrapper
  // eventually this should use the actual data_batch from the scan operator
  std::vector<std::shared_ptr<::cucascade::data_batch>> edge_batches;

  // TODO: once integrated with scan operator, use actual input_data.get_data_batches()
  // right now it builds from buffer manager's cached columns
  // eventually it should integrate with table scan pipeline to receive data as data_batch objects
  BuildCSR(edge_batches, stream);

  auto end      = high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(end - start);
  SIRIUS_LOG_INFO("CSR construction time: {:.2f} ms", duration.count() / 1000.0);
}

// ---------------------------------------------------------------------------
// BuildCSR - runs the three CUDA kernels and stores result in csr_cache
// ---------------------------------------------------------------------------
void GPUCSRConstructionOperator::BuildCSR(
  const std::vector<std::shared_ptr<::cucascade::data_batch>>& edge_batches,
  rmm::cuda_stream_view stream) const
{
  auto& mgr = GPUBufferManager::GetInstance();

  // read from buffer manager directly
  // (once scan integration is complete, extract from edge_batches parameter)
  string up_table = parsed.edge_table;
  std::transform(up_table.begin(), up_table.end(), up_table.begin(), ::toupper);

  string src_col = parsed.edge_src_col.empty() ? "SRC" : parsed.edge_src_col;
  string dst_col = parsed.edge_dst_col.empty() ? "DST" : parsed.edge_dst_col;
  std::transform(src_col.begin(), src_col.end(), src_col.begin(), ::toupper);
  std::transform(dst_col.begin(), dst_col.end(), dst_col.begin(), ::toupper);

  auto& table = mgr.tables[up_table];
  auto src_it = std::find(table->column_names.begin(), table->column_names.end(), src_col);
  auto dst_it = std::find(table->column_names.begin(), table->column_names.end(), dst_col);
  idx_t src_idx = std::distance(table->column_names.begin(), src_it);
  idx_t dst_idx = std::distance(table->column_names.begin(), dst_it);

  // column pointers from buffer manager
  auto* src_ptr           = reinterpret_cast<int64_t*>(table->columns[src_idx]->data_wrapper.data);
  auto* dst_ptr           = reinterpret_cast<int64_t*>(table->columns[dst_idx]->data_wrapper.data);
  const int64_t num_edges = static_cast<int64_t>(table->columns[src_idx]->column_length);

  // determine num_vertices by finding max ID across both columns
  // TODO: replace with GPU reduce kernel to avoid CPU roundtrip
  int64_t max_id = 0;
  {
    std::vector<int64_t> src_host(num_edges), dst_host(num_edges);
    callCudaMemcpyDeviceToHost(src_host.data(), src_ptr, num_edges, 0);
    callCudaMemcpyDeviceToHost(dst_host.data(), dst_ptr, num_edges, 0);
    for (int64_t i = 0; i < num_edges; i++) {
      max_id = std::max(max_id, std::max(src_host[i], dst_host[i]));
    }
  }
  const int64_t num_vertices = max_id + 1;

  SIRIUS_LOG_DEBUG("Building CSR: {} vertices, {} edges", num_vertices, num_edges);

  // allocate CSR arrays in GPU cache (if cached it survives across queries)
  int64_t* degree  = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);
  int64_t* offsets = mgr.customCudaMalloc<int64_t>(num_vertices + 1, 0, /*caching=*/true);
  int64_t* indices = mgr.customCudaMalloc<int64_t>(num_edges, 0, /*caching=*/true);
  cudaMemset(degree, 0, num_vertices * sizeof(int64_t));

  // --- Kernel 1: degree counting ---
  SIRIUS_LOG_DEBUG("Launching degree count kernel");
  LaunchDegreeCountKernel(src_ptr, degree, num_edges, num_vertices);

  // --- Kernel 2: prefix scan (exclusive) → offsets ---
  SIRIUS_LOG_DEBUG("Launching prefix scan kernel");
  // TODO: swap to benchmark
  // LaunchCubExclusiveScan(degree, offsets, num_vertices);
  LaunchPrefixScanKernel(degree, offsets, num_vertices);

  // --- Kernel 3: scatter edges into indices ---
  SIRIUS_LOG_DEBUG("Launching scatter kernel");
  // Scatter needs a mutable copy of offsets as write cursors; reuse degree array
  // as scratch space since it's no longer needed after the prefix scan
  LaunchScatterKernel(src_ptr, dst_ptr, offsets, indices, num_edges, num_vertices);

  // Synchronize stream to ensure kernels complete
  stream.synchronize();

  // Free scratch degree array (not cached)
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(degree), 0);

  // Store in cache
  auto new_csr          = make_shared_ptr<CachedCSR>();
  new_csr->offsets      = offsets;
  new_csr->indices      = indices;
  new_csr->num_vertices = num_vertices;
  new_csr->num_edges    = num_edges;
  new_csr->is_weighted  = false;

  const string key = CachedCSR::MakeKey(parsed.edge_table,
                                        parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col,
                                        parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col);

  mgr.csr_cache[key] = new_csr;
  csr                = new_csr;

  SIRIUS_LOG_INFO("CSR construction complete: {} vertices, {} edges cached at key '{}'",
                  num_vertices,
                  num_edges,
                  key);
}

}  // namespace sirius::op
