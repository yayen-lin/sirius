#include "graph/GPUGraphTraversalOperator.hpp"
#include "graph/GPUCSRConstructionOperator.hpp"  // for build_csr_if_needed

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

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
  SIRIUS_LOG_DEBUG("[GRAPH] GPUGraphTraversalOperator::execute() called, traversal_done={}",
                  traversal_done);
  if (!csr) { throw InternalException("GPUGraphTraversalOperator: CSR is null"); }

  // return cached result on subsequent calls
  if (traversal_done && cached_result) { return std::make_unique<operator_data>(*cached_result); }

  // build CSR here in execute() where a live pipeline-task reservation is held.
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
  SIRIUS_LOG_DEBUG("GPUGraphTraversalOperator traversal time: {:.2f} ms", duration.count() / 1000.0);

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

  auto mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer d_source_buf(src_ids.data(), num_sources * sizeof(int64_t), stream, mr);
  auto* d_source_ids = static_cast<int64_t*>(d_source_buf.data());

  // Launch kernel - returns device_uvectors (cucascade-managed memory)
  auto result = LaunchEdgeTraversalKernel(
    csr->offsets.data(), csr->indices.data(), d_source_ids, num_sources, stream, mr);

  const int64_t result_count = result.node_ids.size();
  std::vector<std::unique_ptr<cudf::column>> columns;

  // Create distance buffer filled with 1s (edge traversal is always distance=1)
  rmm::device_uvector<int64_t> distance_ones(result_count, stream, mr);
  LaunchFillKernel(distance_ones.data(), 1, result_count, stream);

  for (const auto& col_name : parsed.output_columns) {
    if (col_name == "predecessor") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result.predecessor_ids.release(), stream, mr),
        rmm::device_buffer{},
        0));
    } else if (col_name == "distance") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(distance_ones.release(), stream, mr),
        rmm::device_buffer{},
        0));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result.node_ids.release(), stream, mr),
        rmm::device_buffer{},
        0));
    }
  }

  auto result_table = std::make_unique<cudf::table>(std::move(columns));
  auto batch        = sirius::make_data_batch(std::move(result_table), *_gpu_memory_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));

  SIRIUS_LOG_DEBUG("Edge traversal complete: {} results", result_count);
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

  auto mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer d_source_buf(src_ids.data(), num_sources * sizeof(int64_t), stream, mr);
  auto* d_source_ids = static_cast<int64_t*>(d_source_buf.data());

  // Launch kernel - returns device_uvectors (cucascade-managed memory)
  auto result = LaunchBFSKernel(csr->offsets.data(),
                                csr->indices.data(),
                                d_source_ids,
                                num_sources,
                                csr->num_vertices,
                                stream,
                                mr);

  const int64_t result_count = result.node_ids.size();

  // Emit columns in the order declared in COLUMNS (...), using output_columns names.
  std::vector<std::unique_ptr<cudf::column>> columns;

  for (const auto& col_name : parsed.output_columns) {
    if (col_name == "distance") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result.distances.release(), stream, mr),
        rmm::device_buffer{},
        0));
    } else if (col_name == "predecessor") {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result.predecessors.release(), stream, mr),
        rmm::device_buffer{},
        0));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        result_count,
        rmm::device_buffer(result.node_ids.release(), stream, mr),
        rmm::device_buffer{},
        0));
    }
  }

  auto result_table = std::make_unique<cudf::table>(std::move(columns));
  auto batch        = sirius::make_data_batch(std::move(result_table), *_gpu_memory_space);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));

  SIRIUS_LOG_DEBUG("BFS complete: {} results", result_count);
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

}  // namespace sirius::op