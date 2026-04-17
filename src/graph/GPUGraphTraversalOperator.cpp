#include "graph/GPUGraphTraversalOperator.hpp"

#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"
#include "utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <cucascade/data/data_batch.hpp>

#include <chrono>
#include <stdexcept>

namespace sirius::op {

using namespace duckdb;

GPUGraphTraversalOperator::GPUGraphTraversalOperator(vector<LogicalType> types,
                                                     const ParsedGraphQuery& parsed,
                                                     shared_ptr<CachedCSR> csr,
                                                     idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GRAPH_TRAVERSAL, std::move(types), estimated_cardinality),
    parsed(parsed),
    csr(std::move(csr))
{
}

std::string GPUGraphTraversalOperator::params_to_string() const
{
  string op_str;
  switch (parsed.op) {
    case OperationType::EDGE_TRAVERSAL: op_str = "EDGE_TRAVERSAL"; break;
    case OperationType::BFS: op_str = "BFS"; break;
    case OperationType::UNWEIGHTED_SHORTEST_PATH: op_str = "SHORTEST_PATH"; break;
    case OperationType::WEIGHTED_SHORTEST_PATH: op_str = "WEIGHTED_SHORTEST_PATH"; break;
  }
  string sources_str =
    parsed.has_source_filter ? std::to_string(parsed.sourceLiteralIDs().size()) : "all";

  return " [op=" + op_str + ", sources=" + sources_str + "]";
}

std::unique_ptr<operator_data> GPUGraphTraversalOperator::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  if (!csr) { throw InternalException("GPUGraphTraversalOperator: CSR is null"); }

  // return cached result on subsequent calls
  if (traversal_done && cached_result) { return std::make_unique<operator_data>(*cached_result); }

  auto start = std::chrono::high_resolution_clock::now();

  std::unique_ptr<operator_data> result;

  switch (parsed.op) {
    case OperationType::EDGE_TRAVERSAL: result = RunEdgeTraversal(stream); break;
    case OperationType::BFS:
    case OperationType::UNWEIGHTED_SHORTEST_PATH: result = RunBFS(stream); break;
    case OperationType::WEIGHTED_SHORTEST_PATH:
      throw NotImplementedException("Weighted shortest path not yet implemented");
  }

  traversal_done = true;
  cached_result  = std::make_unique<operator_data>(*result);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_INFO("GPUGraphTraversalOperator traversal time: {:.2f} ms", duration.count() / 1000.0);

  return result;
}

std::unique_ptr<operator_data> GPUGraphTraversalOperator::RunEdgeTraversal(
  rmm::cuda_stream_view stream) const
{
  auto& mgr = GPUBufferManager::GetInstance();

  if (!parsed.has_source_filter) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: edge traversal without source filter not yet supported");
  }

  if (!parsed.sourceIsLiteralList()) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: subquery source filter not yet supported");
  }

  const auto& src_ids = parsed.sourceLiteralIDs();
  int64_t num_sources = static_cast<int64_t>(src_ids.size());

  int64_t* d_source_ids = mgr.customCudaMalloc<int64_t>(num_sources, 0, false);
  callCudaMemcpyHostToDevice(d_source_ids, const_cast<int64_t*>(src_ids.data()), num_sources, 0);

  int64_t* result_node_ids        = nullptr;
  int64_t* result_predecessor_ids = nullptr;
  int64_t result_count            = 0;
  LaunchEdgeTraversalKernel(csr->offsets,
                            csr->indices,
                            d_source_ids,
                            num_sources,
                            result_node_ids,
                            result_predecessor_ids,
                            result_count);

  mgr.customCudaFree(reinterpret_cast<uint8_t*>(d_source_ids), 0);

  // convert results to data_batch format
  // TODO: construct cudf columns and wrap in data_batch
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;

  // create a data_batch with the results
  // (requires converting raw GPU pointers to cudf::column objects and then wrapping in cucascade::data_batch)
  SIRIUS_LOG_INFO("Edge traversal complete: {} results", result_count);

  return std::make_unique<operator_data>();
}

std::unique_ptr<operator_data> GPUGraphTraversalOperator::RunBFS(rmm::cuda_stream_view stream) const
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

  int64_t* d_source_ids = mgr.customCudaMalloc<int64_t>(num_sources, 0, false);
  callCudaMemcpyHostToDevice(d_source_ids, const_cast<int64_t*>(src_ids.data()), num_sources, 0);

  int64_t* result_node_ids        = nullptr;
  int64_t* result_distances       = nullptr;
  int64_t* result_predecessor_ids = nullptr;
  int64_t result_count            = 0;
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

  // convert results to data_batch format
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;

  SIRIUS_LOG_INFO("BFS complete: {} results", result_count);

  return std::make_unique<operator_data>();
}

}  // namespace sirius::op