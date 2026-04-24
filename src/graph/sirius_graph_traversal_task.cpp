#include "graph/sirius_graph_traversal_task.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

namespace sirius::graph {

sirius_graph_traversal_task::sirius_graph_traversal_task(
  uint64_t task_id,
  std::vector<cucascade::shared_data_repository*> data_repos,
  std::unique_ptr<pipeline::sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<pipeline::sirius_pipeline_task_global_state> global_state,
  duckdb::shared_ptr<duckdb::sirius_cached_csr> csr)
  : pipeline::gpu_pipeline_task(
      task_id, std::move(data_repos), std::move(local_state), std::move(global_state)),
    _csr(std::move(csr))
{
}

std::size_t sirius_graph_traversal_task::get_estimated_reservation_size() const
{
  if (!_csr) {
    SIRIUS_LOG_WARN("graph_traversal_task: CSR pointer is null, using fallback estimate");
    return 1024 * 1024 * 100;  // 100 MB
  }

  std::size_t estimated_bytes = 0;

  // csr is already built and cached, tracked by cucascade
  if (_csr->is_built) {
    // estimate output size: worst case is all vertices are reachable
    const int64_t max_output_vertices = _csr->num_vertices;
    const std::size_t output_arrays   = 3;  // node_ids, distances, predecessors

    // bfs output: max_output_vertices * 3 arrays * sizeof(int64_t)
    estimated_bytes = max_output_vertices * output_arrays * sizeof(int64_t);

    // add overhead for temporary buffers (visited array, frontiers, etc.)
    const std::size_t temp_buffers = _csr->num_vertices * 2 * sizeof(int64_t);
    estimated_bytes += temp_buffers;

    SIRIUS_LOG_DEBUG(
      "graph_traversal_task: CSR cached, estimated {} bytes ({:.2f} MB) for {} vertices",
      estimated_bytes,
      static_cast<double>(estimated_bytes) / (1024.0 * 1024.0),
      _csr->num_vertices);
  } else {
    // CSR not yet built, need memory for:
    // 1. csr construction (offsets + indices)
    // 2. traversal output
    // 3. temporary buffers

    // calculate actual edge count from pending batches
    int64_t num_edges = 0;
    for (const auto& batch : _csr->pending_batches) {
      if (batch && batch->get_data()) {
        auto tbl = sirius::get_cudf_table_view(*batch);
        num_edges += tbl.num_rows();
      }
    }

    // Conservative estimate for vertex count:
    // - Social networks: avg_degree ~10-50 (sparse)
    // - Dense graphs: avg_degree can be much higher
    // - Use conservative multiplier: num_vertices ≈ num_edges / 2 (assumes avg_degree ~2)
    // - Add safety margin since graph can have isolated vertices
    const int64_t estimated_vertices = (num_edges / 2) * 3;  // 3x safety margin

    // csr construction:
    // - offsets: (num_vertices + 1) * sizeof(int64_t)
    // - indices: num_edges * sizeof(int64_t)
    // - degree array (temporary): num_vertices * sizeof(int64_t)
    const std::size_t csr_offsets = (estimated_vertices + 1) * sizeof(int64_t);
    const std::size_t csr_indices = num_edges * sizeof(int64_t);
    const std::size_t csr_temps   = estimated_vertices * sizeof(int64_t);  // degree array
    const std::size_t csr_size    = csr_offsets + csr_indices + csr_temps;

    // traversal output (worst case: all vertices reachable)
    // - node_ids, distances, predecessors: 3 arrays
    const std::size_t output_size = estimated_vertices * 3 * sizeof(int64_t);

    // traversal temporary buffers (visited, frontiers, etc.)
    const std::size_t traversal_temps = estimated_vertices * 2 * sizeof(int64_t);

    // input edge batches need to be materialized for concatenation
    std::size_t input_batch_size = 0;
    for (const auto& batch : _csr->pending_batches) {
      if (batch && batch->get_data()) {
        input_batch_size += batch->get_data()->get_uncompressed_data_size_in_bytes();
      }
    }

    estimated_bytes = csr_size + output_size + traversal_temps + input_batch_size;

    SIRIUS_LOG_DEBUG(
      "graph_traversal_task: CSR not built, estimated {} bytes ({:.2f} MB) "
      "for {} edges, ~{} vertices (est), input batches: {:.2f} MB",
      estimated_bytes,
      static_cast<double>(estimated_bytes) / (1024.0 * 1024.0),
      num_edges,
      estimated_vertices,
      static_cast<double>(input_batch_size) / (1024.0 * 1024.0));
  }

  // Use memory history to refine estimate if available
  auto& global = this->_global_state->cast<pipeline::gpu_pipeline_task_global_state>();
  auto input_basis =
    _local_state->cast<pipeline::gpu_pipeline_task_local_state>().get_task_consumption_basis();
  auto refined = global.get_memory_history().estimate_peak_memory(input_basis);
  if (refined && *refined > estimated_bytes) {
    SIRIUS_LOG_DEBUG(
      "graph_traversal_task: memory history refined estimate to {} bytes ({:.2f} MB)",
      *refined,
      static_cast<double>(*refined) / (1024.0 * 1024.0));
    return *refined;
  }

  return estimated_bytes;
}

std::unique_ptr<pipeline::gpu_pipeline_task> sirius_graph_traversal_task::create_rescheduled_task(
  uint64_t task_id, std::unique_ptr<pipeline::sirius_pipeline_task_local_state> local_state)
{
  return std::make_unique<sirius_graph_traversal_task>(
    task_id, get_data_repos(), std::move(local_state), get_shared_global_state(), _csr);
}

}  // namespace sirius::graph
