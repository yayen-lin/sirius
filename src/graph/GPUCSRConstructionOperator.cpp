#include "graph/GPUCSRConstructionOperator.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda_runtime.h>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace sirius::op {

using namespace duckdb;

// build_csr_if_needed — free function, called from GPUGraphTraversalOperator::execute()
// while a live pipeline-task reservation is held.
void build_csr_if_needed(shared_ptr<CachedCSR>& csr, rmm::cuda_stream_view stream)
{
  if (csr->is_built) { return; }

  SIRIUS_LOG_INFO("[GRAPH] build_csr_if_needed: starting CSR build from {} pending batches",
                  csr->pending_batches.size());

  auto start = std::chrono::high_resolution_clock::now();

  if (csr->pending_batches.empty()) {
    throw InvalidInputException("build_csr_if_needed: no edge data in pending_batches");
  }

  // extract src (col 0) and dst (col 1) column views from each batch
  std::vector<cudf::column_view> src_views, dst_views;
  for (auto& batch : csr->pending_batches) {
    auto tbl = sirius::get_cudf_table_view(*batch);
    src_views.push_back(tbl.column(0));
    dst_views.push_back(tbl.column(1));
  }

  // Use cuda_memory_resource to bypass the cucascade reservation adaptor for cudf
  // output allocations (internal scratch still avoids it via thrust below).
  static rmm::mr::cuda_memory_resource cuda_mr;

  SIRIUS_LOG_INFO("[GRAPH] build_csr_if_needed: concatenating edge columns");
  auto src_concat = cudf::concatenate(src_views, stream, cuda_mr);
  auto dst_concat = cudf::concatenate(dst_views, stream, cuda_mr);

  const int64_t num_edges = src_concat->size();
  const auto* src_ptr     = src_concat->view().data<int64_t>();
  const auto* dst_ptr     = dst_concat->view().data<int64_t>();

  SIRIUS_LOG_INFO("[GRAPH] build_csr_if_needed: concatenate done, num_edges={}", num_edges);

  // find num_vertices — LaunchFindMaxKernel uses thrust in a .cu file,
  // so it allocates via CUB (cudaMalloc) and never touches the cucascade adaptor
  int64_t src_max            = LaunchFindMaxKernel(src_ptr, num_edges);
  int64_t dst_max            = LaunchFindMaxKernel(dst_ptr, num_edges);
  const int64_t num_vertices = std::max(src_max, dst_max) + 1;

  SIRIUS_LOG_INFO("[GRAPH] build_csr_if_needed: num_vertices={}", num_vertices);

  // allocate CSR arrays with cudaMalloc (outside the cucascade pool)
  int64_t* degree  = nullptr;
  int64_t* offsets = nullptr;
  int64_t* indices = nullptr;
  cudaMalloc(&degree, num_vertices * sizeof(int64_t));
  cudaMalloc(&offsets, (num_vertices + 1) * sizeof(int64_t));
  cudaMalloc(&indices, num_edges * sizeof(int64_t));

  cudaMemset(degree, 0, num_vertices * sizeof(int64_t));
  LaunchDegreeCountKernel(src_ptr, degree, num_edges, num_vertices);
  LaunchPrefixScanKernel(degree, offsets, num_vertices);
  LaunchScatterKernel(src_ptr, dst_ptr, offsets, indices, num_edges, num_vertices);
  cudaDeviceSynchronize();
  cudaFree(degree);

  csr->offsets      = offsets;
  csr->indices      = indices;
  csr->num_vertices = num_vertices;
  csr->num_edges    = num_edges;
  csr->is_weighted  = false;
  csr->is_built     = true;

  // release the raw edge batches now that we no longer need them
  csr->pending_batches.clear();

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_INFO("[GRAPH] build_csr_if_needed: CSR built — {} vertices, {} edges ({:.2f} ms)",
                  num_vertices,
                  num_edges,
                  duration.count() / 1000.0);
}

GPUCSRConstructionOperator::GPUCSRConstructionOperator(
  vector<LogicalType> types,
  const ParsedGraphQuery& parsed,
  idx_t estimated_cardinality,
  cucascade::memory::memory_space& gpu_memory_space)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CSR_CONSTRUCTION, std::move(types), estimated_cardinality),
    parsed(parsed),
    _gpu_memory_space(&gpu_memory_space)
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
  SIRIUS_LOG_INFO("[GRAPH] CSR execute() called");

  auto* p = dynamic_cast<const pipelineable_operator_data*>(&input_data);
  if (p) {
    for (auto& batch : p->get_data_batches()) {
      if (batch) { csr->pending_batches.push_back(batch); }
    }
    SIRIUS_LOG_DEBUG("[GRAPH] CSR execute(): pending_batches now={}", csr->pending_batches.size());
  }

  // Return a 0-row trigger batch so the pipeline infrastructure pushes something
  // to GRAPH_TRAVERSAL's repo. Without this the task_creator sees an empty repo
  // and never creates a task for Pipeline #2, causing a hang.
  // GRAPH_TRAVERSAL ignores input_data entirely; it reads from the shared csr pointer.
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
  auto trigger_table = std::make_unique<cudf::table>(std::move(cols));
  auto trigger_batch = sirius::make_data_batch(std::move(trigger_table), *_gpu_memory_space);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{std::move(trigger_batch)});
}

void GPUCSRConstructionOperator::finalize_operator()
{
  // GPU work is deferred to GPUGraphTraversalOperator::execute() where a live
  // pipeline-task reservation is held. Nothing to do here.
  SIRIUS_LOG_INFO("[GRAPH] CSR finalize_operator(): pending_batches={}, deferred to traversal",
                  csr->pending_batches.size());
}

}  // namespace sirius::op
