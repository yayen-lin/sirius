#include "graph/GPUCSRConstructionOperator.hpp"

#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace duckdb {

GPUCSRConstructionOperator::GPUCSRConstructionOperator(vector<LogicalType> types,
                                                       const ParsedGraphQuery& parsed,
                                                       idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::INVALID, std::move(types), estimated_cardinality),
    parsed(parsed)
{
}

// ---------------------------------------------------------------------------
// Sink - receives the edge columns from GPUPhysicalTableScan
// We just accumulate into edge_relation; actual CSR construction happens in
// CombineFinalize once all input is available.
// ---------------------------------------------------------------------------
SinkResultType GPUCSRConstructionOperator::Sink(GPUIntermediateRelation& input_relation) const
{
  // Store the incoming edge data - column 0 is src, column 1 is dst
  // (column order matches what GPUGraphFunction requests from the table scan)

  // For now accumulate the first batch; batching across multiple Sink calls
  // will be needed once chunked input is wired up
  // TODO: handle multi-chunk input (append columns rather than overwrite)

  return SinkResultType::NEED_MORE_INPUT;
}

// ---------------------------------------------------------------------------
// CombineFinalize - build the CSR
// ---------------------------------------------------------------------------
SinkFinalizeType GPUCSRConstructionOperator::CombineFinalize(
  vector<shared_ptr<GPUIntermediateRelation>>& input, GPUIntermediateRelation& output) const
{
  auto start = std::chrono::high_resolution_clock::now();

  auto& mgr        = GPUBufferManager::GetInstance();
  const string key = CachedCSR::MakeKey(parsed.edge_table,
                                        parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col,
                                        parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col);

  // Cache hit - reuse existing CSR, skip construction entirely
  auto it = mgr.csr_cache.find(key);
  if (it != mgr.csr_cache.end()) {
    csr = it->second;
    SIRIUS_LOG_INFO("CSR cache hit for key '{}', skipping construction", key);
    return SinkFinalizeType::READY;
  }

  // Cache miss - read edge columns directly from buffer manager
  // (edge table was cached via a prior gpu_processing call)
  SIRIUS_LOG_INFO("CSR cache miss for key '{}', building CSR", key);

  string up_table = parsed.edge_table;
  std::transform(up_table.begin(), up_table.end(), up_table.begin(), ::toupper);

  string src_col = parsed.edge_src_col.empty() ? "SRC" : parsed.edge_src_col;
  string dst_col = parsed.edge_dst_col.empty() ? "DST" : parsed.edge_dst_col;
  std::transform(src_col.begin(), src_col.end(), src_col.begin(), ::toupper);
  std::transform(dst_col.begin(), dst_col.end(), dst_col.begin(), ::toupper);

  if (!mgr.checkIfColumnCached(up_table, src_col) || !mgr.checkIfColumnCached(up_table, dst_col)) {
    throw InvalidInputException(
      "Edge table '{}' with columns '{}'/'{}' not cached. "
      "Call gpu_processing('SELECT * FROM {}') first.",
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

  GPUIntermediateRelation edge_relation(2);
  edge_relation.columns[0] = table->columns[src_idx];
  edge_relation.columns[1] = table->columns[dst_idx];

  BuildCSR(edge_relation);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_INFO("CSR construction time: {:.2f} ms", duration.count() / 1000.0);

  return SinkFinalizeType::READY;
}

// ---------------------------------------------------------------------------
// BuildCSR - runs the three CUDA kernels and stores result in csr_cache
// ---------------------------------------------------------------------------
void GPUCSRConstructionOperator::BuildCSR(GPUIntermediateRelation& edge_relation) const
{
  auto& mgr = GPUBufferManager::GetInstance();

  // Column 0 = src, column 1 = dst (int64)
  auto* src_col           = reinterpret_cast<int64_t*>(edge_relation.columns[0]->data_wrapper.data);
  auto* dst_col           = reinterpret_cast<int64_t*>(edge_relation.columns[1]->data_wrapper.data);
  const int64_t num_edges = static_cast<int64_t>(edge_relation.columns[0]->column_length);

  // Determine num_vertices by finding max ID across both columns
  // TODO: replace with GPU reduce kernel for large graphs
  int64_t max_id = 0;
  // (CPU fallback for now - acceptable for correctness, optimise later)
  // TODO: LaunchMaxReduceKernel(src_col, dst_col, num_edges, &max_id);
  {
    std::vector<int64_t> src_host(num_edges), dst_host(num_edges);
    callCudaMemcpyDeviceToHost(src_host.data(), src_col, num_edges, 0);
    callCudaMemcpyDeviceToHost(dst_host.data(), dst_col, num_edges, 0);
    for (int64_t i = 0; i < num_edges; i++) {
      max_id = std::max(max_id, std::max(src_host[i], dst_host[i]));
    }
  }
  const int64_t num_vertices = max_id + 1;

  // Allocate CSR arrays in GPU cache (caching=true → survives across queries)
  int64_t* degree  = mgr.customCudaMalloc<int64_t>(num_vertices, 0, /*caching=*/false);
  int64_t* offsets = mgr.customCudaMalloc<int64_t>(num_vertices + 1, 0, /*caching=*/true);
  int64_t* indices = mgr.customCudaMalloc<int64_t>(num_edges, 0, /*caching=*/true);

  // Zero-initialise degree array
  // TODO: replace with cudaMemset kernel, memory leak?
  callCudaMemcpyHostToDevice(degree, new int64_t[num_vertices]{}, num_vertices, 0);

  // --- Kernel 1: degree counting ---
  LaunchDegreeCountKernel(src_col, degree, num_edges, num_vertices);

  // --- Kernel 2: prefix scan (exclusive) → offsets ---
  // To benchmark against cub, swap this call with:
  //   LaunchCubExclusiveScan(degree, offsets, num_vertices);
  LaunchPrefixScanKernel(degree, offsets, num_vertices);

  // --- Kernel 3: scatter edges into indices ---
  // Scatter needs a mutable copy of offsets as write cursors; reuse degree array
  // as scratch space since it's no longer needed after the prefix scan
  LaunchScatterKernel(src_col, dst_col, offsets, indices, num_edges, num_vertices);

  // Free scratch degree array (not cached)
  mgr.customCudaFree(reinterpret_cast<uint8_t*>(degree), 0);

  // Store in cache
  auto new_csr          = make_shared_ptr<CachedCSR>();
  new_csr->offsets      = offsets;
  new_csr->indices      = indices;
  new_csr->num_vertices = num_vertices;
  new_csr->num_edges    = num_edges;
  new_csr->is_weighted  = false;

  const string key   = CachedCSR::MakeKey(parsed.edge_table,
                                        parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col,
                                        parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col);
  mgr.csr_cache[key] = new_csr;
  csr                = new_csr;
}

// ---------------------------------------------------------------------------
// GetData - passes the CachedCSR pointer to GPUGraphTraversalOperator
// The output_relation here is unconventional - we store the CSR pointer in
// column 0's data_wrapper.data as a raw pointer. The traversal operator
// knows to interpret it this way.
// TODO: define a cleaner inter-operator contract once traversal is implemented
// ---------------------------------------------------------------------------
SourceResultType GPUCSRConstructionOperator::GetData(GPUIntermediateRelation& output_relation) const
{
  if (!csr) {
    throw InternalException("GPUCSRConstructionOperator::GetData called before CSR was built");
  }
  // Pass the CSR pointer downstream as a single INT64 column.
  // GPUGraphTraversalOperator recovers it via reinterpret_cast<CachedCSR*>.
  output_relation.columns[0] = make_shared_ptr<GPUColumn>(
    1, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(csr.get()), nullptr);

  SIRIUS_LOG_DEBUG("GPUCSRConstructionOperator::GetData passing CSR ({} vertices, {} edges)",
                   csr->num_vertices,
                   csr->num_edges);
  return SourceResultType::FINISHED;
}

}  // namespace duckdb