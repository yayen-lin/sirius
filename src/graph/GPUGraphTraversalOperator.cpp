#include "graph/GPUGraphTraversalOperator.hpp"

#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "utils.hpp"

#include <chrono>
#include <stdexcept>

namespace duckdb {

GPUGraphTraversalOperator::GPUGraphTraversalOperator(vector<LogicalType> types,
                                                     const ParsedGraphQuery& parsed,
                                                     shared_ptr<CachedCSR> csr,
                                                     idx_t estimated_cardinality)
  : GPUPhysicalOperator(PhysicalOperatorType::INVALID, std::move(types), estimated_cardinality),
    parsed(parsed),
    csr(std::move(csr))
{
}

// ---------------------------------------------------------------------------
// GetData - runs traversal on first call, streams result on subsequent calls
// ---------------------------------------------------------------------------
SourceResultType GPUGraphTraversalOperator::GetData(GPUIntermediateRelation& output_relation) const
{
  if (!csr) { throw InternalException("GPUGraphTraversalOperator: CSR is null"); }

  if (!traversal_done) {
    auto start = std::chrono::high_resolution_clock::now();

    switch (parsed.op) {
      case OperationType::EDGE_TRAVERSAL: RunEdgeTraversal(output_relation); break;
      case OperationType::BFS:
      case OperationType::UNWEIGHTED_SHORTEST_PATH: RunBFS(output_relation); break;
      case OperationType::WEIGHTED_SHORTEST_PATH:
        throw NotImplementedException("Weighted shortest path not yet implemented");
    }

    traversal_done = true;

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("GPUGraphTraversalOperator traversal time: {:.2f} ms",
                    duration.count() / 1000.0);
  }

  return SourceResultType::FINISHED;
}

// ---------------------------------------------------------------------------
// RunEdgeTraversal - 1-hop direct neighbor lookup
// ---------------------------------------------------------------------------
void GPUGraphTraversalOperator::RunEdgeTraversal(GPUIntermediateRelation& output_relation) const
{
  auto& mgr = GPUBufferManager::GetInstance();

  // Extract source IDs from the parsed query filter
  if (!parsed.has_source_filter) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: edge traversal without source filter not yet supported");
  }

  if (!parsed.sourceIsLiteralList()) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: subquery source filter not yet supported");
  }

  const auto& src_ids = parsed.sourceLiteralIDs();

  // Copy source IDs to GPU
  int64_t num_sources   = static_cast<int64_t>(src_ids.size());
  int64_t* d_source_ids = mgr.customCudaMalloc<int64_t>(num_sources, 0, /*caching=*/false);
  callCudaMemcpyHostToDevice(d_source_ids, const_cast<int64_t*>(src_ids.data()), num_sources, 0);

  // Launch kernel - results allocated inside LaunchEdgeTraversalKernel
  LaunchEdgeTraversalKernel(csr->offsets,
                            csr->indices,
                            d_source_ids,
                            num_sources,
                            result_node_ids,
                            result_predecessor_ids,
                            result_count);

  mgr.customCudaFree(reinterpret_cast<uint8_t*>(d_source_ids), 0);

  // Write results into output_relation based on requested output columns
  for (idx_t col_idx = 0; col_idx < parsed.output_columns.size(); col_idx++) {
    const auto& col = parsed.output_columns[col_idx];

    if (col == "distance") {
      // Edge traversal is always distance=1 - allocate and fill a constant column
      int64_t* d_distance = mgr.customCudaMalloc<int64_t>(result_count, 0, /*caching=*/false);
      // TODO: replace with a GPU fill kernel
      std::vector<int64_t> h_distance(result_count, 1);
      callCudaMemcpyHostToDevice(d_distance, h_distance.data(), result_count, 0);
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(d_distance),
                                   nullptr);
    } else if (col == "predecessor") {
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(result_predecessor_ids),
                                   nullptr);
    } else {
      // dst.id or any other node column maps to result_node_ids
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(result_node_ids),
                                   nullptr);
    }
  }
}

// ---------------------------------------------------------------------------
// Breadth first search
// ---------------------------------------------------------------------------
void GPUGraphTraversalOperator::RunBFS(GPUIntermediateRelation& output_relation) const
{
  auto& mgr = GPUBufferManager::GetInstance();

  if (!parsed.has_source_filter) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: BFS without source filter not yet supported");
  }

  if (!parsed.sourceIsLiteralList()) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: subquery source filter not yet supported");
  }

  const auto& src_ids = parsed.sourceLiteralIDs();
  int64_t num_sources = static_cast<int64_t>(src_ids.size());

  // Copy source IDs to GPU
  int64_t* d_source_ids = mgr.customCudaMalloc<int64_t>(num_sources, 0, /*caching=*/false);
  callCudaMemcpyHostToDevice(d_source_ids, const_cast<int64_t*>(src_ids.data()), num_sources, 0);

  LaunchBFSKernel(csr->offsets,
                  csr->indices,
                  d_source_ids,
                  num_sources,
                  csr->num_vertices,
                  result_node_ids,
                  result_distances,
                  result_predecessor_ids,
                  result_count);

  mgr.customCudaFree(reinterpret_cast<uint8_t*>(d_source_ids), 0);

  // Write results into output_relation based on requested output columns
  for (idx_t col_idx = 0; col_idx < parsed.output_columns.size(); col_idx++) {
    const auto& col = parsed.output_columns[col_idx];

    if (col == "distance") {
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(result_distances),
                                   nullptr);
    } else if (col == "predecessor") {
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(result_predecessor_ids),
                                   nullptr);
    } else if (col == "path") {
      // TODO: implement path reconstruction
      throw NotImplementedException("Path reconstruction not yet implemented");
    } else {
      // dst.id or any other node column
      output_relation.columns[col_idx] =
        make_shared_ptr<GPUColumn>(result_count,
                                   GPUColumnType(GPUColumnTypeId::INT64),
                                   reinterpret_cast<uint8_t*>(result_node_ids),
                                   nullptr);
    }
  }
}

}  // namespace duckdb