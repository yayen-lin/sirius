#include "graph/GPUGraphTraversalOperator.hpp"
#include "graph/GPUCSRConstructionOperator.hpp"  // for build_csr_if_needed

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda_runtime.h>

#include <cucascade/data/data_batch.hpp>

#include <chrono>
#include <stdexcept>

namespace sirius::op {

using namespace duckdb;

GPUGraphTraversalOperator::GPUGraphTraversalOperator(
  vector<LogicalType> types,
  const ParsedGraphQuery& parsed,
  shared_ptr<CachedCSR> csr,
  idx_t estimated_cardinality,
  cucascade::memory::memory_space& gpu_memory_space)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GRAPH_TRAVERSAL, std::move(types), estimated_cardinality),
    parsed(parsed),
    csr(std::move(csr)),
    _gpu_memory_space(&gpu_memory_space)
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
  SIRIUS_LOG_INFO("[GRAPH] GPUGraphTraversalOperator::execute() called, traversal_done={}",
                  traversal_done);
  spdlog::default_logger()->flush();
  if (!csr) { throw InternalException("GPUGraphTraversalOperator: CSR is null"); }

  // return cached result on subsequent calls
  if (traversal_done && cached_result) { return std::make_unique<operator_data>(*cached_result); }

  // Build CSR here, inside execute() where a live pipeline-task reservation is held.
  // This avoids the 0-byte-budget OOM that occurs when building in finalize_operator().
  build_csr_if_needed(csr, stream);

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
  if (!parsed.has_source_filter) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: edge traversal without source filter not yet supported");
  }

  if (!parsed.sourceIsLiteralList()) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: subquery source filter not yet supported");
  }

  const auto& src_ids    = parsed.sourceLiteralIDs();
  const auto num_sources = static_cast<int64_t>(src_ids.size());

  static rmm::mr::cuda_memory_resource cuda_mr;
  rmm::device_buffer d_source_buf(src_ids.data(), num_sources * sizeof(int64_t), stream, &cuda_mr);
  auto* d_source_ids = static_cast<int64_t*>(d_source_buf.data());

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

  // Emit columns in the order declared in COLUMNS (...), using output_columns names.
  // "distance" for direct edge traversal is always 1 (one hop).
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<int64_t> h_ones(result_count, 1);  // constant distance=1 buffer for edge traversal

  for (const auto& col_name : parsed.output_columns) {
    if (col_name == "predecessor") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result_predecessor_ids, result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    } else if (col_name == "distance") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(h_ones.data(), result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result_node_ids, result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    }
  }

  cudaFree(result_node_ids);
  cudaFree(result_predecessor_ids);

  auto result_table = std::make_unique<cudf::table>(std::move(columns));
  auto batch        = sirius::make_data_batch(std::move(result_table), *_gpu_memory_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));

  SIRIUS_LOG_INFO("Edge traversal complete: {} results", result_count);
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<operator_data> GPUGraphTraversalOperator::RunBFS(rmm::cuda_stream_view stream) const
{
  if (!parsed.has_source_filter) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: BFS without source filter not yet supported");
  }

  if (!parsed.sourceIsLiteralList()) {
    throw NotImplementedException(
      "GPUGraphTraversalOperator: subquery source filter not yet supported");
  }

  const auto& src_ids    = parsed.sourceLiteralIDs();
  const auto num_sources = static_cast<int64_t>(src_ids.size());

  static rmm::mr::cuda_memory_resource cuda_mr;
  rmm::device_buffer d_source_buf(src_ids.data(), num_sources * sizeof(int64_t), stream, &cuda_mr);
  auto* d_source_ids = reinterpret_cast<int64_t*>(d_source_buf.data());

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

  // Emit columns in the order declared in COLUMNS (...), using output_columns names.
  std::vector<std::unique_ptr<cudf::column>> columns;

  for (const auto& col_name : parsed.output_columns) {
    if (col_name == "distance") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result_distances, result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    } else if (col_name == "predecessor") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result_predecessor_ids, result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result_node_ids, result_count * sizeof(int64_t), stream, &cuda_mr),
        rmm::device_buffer{},
        0));
    }
  }

  cudaFree(result_node_ids);
  cudaFree(result_distances);
  cudaFree(result_predecessor_ids);

  auto result_table = std::make_unique<cudf::table>(std::move(columns));
  auto batch        = sirius::make_data_batch(std::move(result_table), *_gpu_memory_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));

  SIRIUS_LOG_INFO("BFS complete: {} results", result_count);
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op