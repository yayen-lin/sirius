#include "graph/sirius_physical_csr_construction.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda_runtime.h>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <chrono>
#include <mutex>

namespace sirius::op {

using namespace duckdb;

void build_csr_if_needed(shared_ptr<sirius_cached_csr>& csr, rmm::cuda_stream_view stream)
{
  // check without lock
  if (csr->is_built) { return; }
  std::scoped_lock lock(csr->build_mutex);

  // another thread may have built it while we waited for the lock
  if (csr->is_built) {
    SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: CSR already built by another thread");
    return;
  }

  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: starting CSR build from {} pending batches",
                  csr->pending_batches.size());

  auto start = std::chrono::high_resolution_clock::now();

  if (csr->pending_batches.empty()) {
    throw InvalidInputException("build_csr_if_needed: no edge data in pending_batches");
  }

  // extract src and dst column views from each batch
  std::vector<cudf::column_view> src_views, dst_views;
  for (auto& batch : csr->pending_batches) {
    auto tbl = sirius::get_cudf_table_view(*batch);
    src_views.push_back(tbl.column(0)); // src
    dst_views.push_back(tbl.column(1)); // dst
  }

  // use cuda_memory_resource for cudf operations (bypass cucascade for cudf's internal allocations)
  // this keeps data pinned in gpu memory (no spilling)
  static rmm::mr::cuda_memory_resource cuda_mr;
  // use current device resource for our kernel allocations (cucascade-managed)
  auto mr = cudf::get_current_device_resource_ref();

  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: concatenating edge columns");
  auto src_concat = cudf::concatenate(src_views, stream, cuda_mr);
  auto dst_concat = cudf::concatenate(dst_views, stream, cuda_mr);

  const int64_t num_edges = src_concat->size();
  const auto* src_ptr     = src_concat->view().data<int64_t>();
  const auto* dst_ptr     = dst_concat->view().data<int64_t>();

  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: concatenate done, num_edges={}", num_edges);

  // find num_vertices
  int64_t src_max            = LaunchFindMaxKernel(src_ptr, num_edges);
  int64_t dst_max            = LaunchFindMaxKernel(dst_ptr, num_edges);
  const int64_t num_vertices = std::max(src_max, dst_max) + 1;

  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: num_vertices={}", num_vertices);

  // allocate sorted edge arrays via RMM for degree counting
  rmm::device_uvector<int64_t> src_sorted(num_edges, stream);
  rmm::device_uvector<int64_t> dst_sorted(num_edges, stream);

  // sort edges by source vertex
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: LaunchEdgeSortKernel");
  LaunchEdgeSortKernel(src_ptr, dst_ptr, src_sorted.data(), dst_sorted.data(), num_edges, stream);

  // allocate CSR arrays via RMM (device_uvector owns the memory and frees it on destruction)
  rmm::device_uvector<int64_t> degree_vec(num_vertices, stream);
  rmm::device_uvector<int64_t> offsets_vec(num_vertices + 1, stream);
  rmm::device_uvector<int64_t> indices_vec(num_edges, stream);

  cudaMemsetAsync(degree_vec.data(), 0, num_vertices * sizeof(int64_t), stream.value());
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: LaunchDegreeCountKernel");
  LaunchDegreeCountKernel(src_sorted.data(), degree_vec.data(), num_edges, num_vertices);
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: LaunchPrefixScanKernel");
  LaunchPrefixScanKernel(degree_vec.data(), offsets_vec.data(), num_vertices, stream, mr);
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: LaunchScatterKernel");
  LaunchScatterKernel(src_sorted.data(), dst_sorted.data(), offsets_vec.data(), indices_vec.data(), num_edges, num_vertices, stream, mr);
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: cudaDeviceSynchronize");
  cudaDeviceSynchronize(); // degree_vec freed automatically here

  csr->offsets      = std::move(offsets_vec);
  csr->indices      = std::move(indices_vec);
  csr->num_vertices = num_vertices;
  csr->num_edges    = num_edges;
  csr->is_weighted  = false; // weighted operation not supported yet
  csr->is_built     = true;

  // store in cache
  if (!csr->key.empty()) { sirius_csr_cache::instance().insert(csr->key, csr); }

  // clear edge batches
  csr->pending_batches.clear();

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("[GRAPH] build_csr_if_needed: CSR built — {} vertices, {} edges ({:.2f} ms)",
                  num_vertices,
                  num_edges,
                  duration.count() / 1000.0);
}

sirius_physical_csr_construction::sirius_physical_csr_construction(
  vector<LogicalType> types,
  const sirius_parsed_graph_query& parsed,
  idx_t estimated_cardinality,
  cucascade::memory::memory_space& gpu_memory_space)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::CSR_CONSTRUCTION, std::move(types), estimated_cardinality),
    parsed(parsed),
    _gpu_memory_space(&gpu_memory_space)
{
}

std::string sirius_physical_csr_construction::params_to_string() const
{
  return " [edge_table=" + parsed.edge_table +
         ", src=" + (parsed.edge_src_col.empty() ? "src" : parsed.edge_src_col) +
         ", dst=" + (parsed.edge_dst_col.empty() ? "dst" : parsed.edge_dst_col) + "]";
}

std::unique_ptr<operator_data> sirius_physical_csr_construction::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  SIRIUS_LOG_DEBUG("[GRAPH] CSR execute() called, csr->is_built={}", csr->is_built);

  // non-cached path, accumulate edge batches for CSR construction on the first build
  if (!csr->is_built) {
    auto* p = dynamic_cast<const pipelineable_operator_data*>(&input_data);
    if (p) {
      for (auto& batch : p->get_data_batches()) {
        if (batch) { csr->pending_batches.push_back(batch); }
      }
      SIRIUS_LOG_DEBUG("[GRAPH] CSR execute(): pending_batches now={}",
                       csr->pending_batches.size());
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

  // cached path, CSR already built, return empty result
  else {
    // finalize_operator() will push the trigger to GRAPH_TRAVERSAL
    SIRIUS_LOG_DEBUG("[GRAPH] CSR execute(): CSR already built (cached), returning empty result");
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{});
  }
}

void sirius_physical_csr_construction::finalize_operator()
{
  // non-cached path, GPU build is deferred to sirius_physical_graph_traversal::execute() where a
  // live pipeline-task reservation is held
  if (!csr->is_built) {
    SIRIUS_LOG_DEBUG("[GRAPH] CSR finalize_operator(): pending_batches={}, deferred to traversal",
                    csr->pending_batches.size());
    return;
  }

  // cached path, execute() was never called; push the trigger directly to GRAPH_TRAVERSAL.
  SIRIUS_LOG_DEBUG(
    "[GRAPH] CSR finalize_operator(): cached CSR, pushing trigger to GRAPH_TRAVERSAL");
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
  auto trigger_table = std::make_unique<cudf::table>(std::move(cols));
  auto trigger_batch = sirius::make_data_batch(std::move(trigger_table), *_gpu_memory_space);
  auto trigger_ptr   = std::shared_ptr<::cucascade::data_batch>(std::move(trigger_batch));

  for (auto& next_port : get_next_port_after_sink()) {
    next_port.next_operator->push_data_batch(next_port.next_operator_port_name, trigger_ptr);
  }
}

void sirius_physical_csr_construction::build_pipelines(pipeline::sirius_pipeline& current,
                                                 pipeline::sirius_meta_pipeline& meta_pipeline)
{
  SIRIUS_LOG_DEBUG("[GRAPH] CSR build_pipelines ENTRY: csr={}, csr->is_built={}, children.size()={}",
                  static_cast<void*>(csr.get()),
                  csr ? csr->is_built : false,
                  children.size());
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // build child pipeline (scan for non-cached path, dummy scan if already cached)
  D_ASSERT(!children.empty());
  auto& child_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta.build(*children[0]);
  SIRIUS_LOG_DEBUG("[GRAPH] CSR build_pipelines: built child pipeline, cached={}, child source={}",
                  csr ? csr->is_built : false,
                  child_meta.get_base_pipeline()->get_source()
                    ? child_meta.get_base_pipeline()->get_source()->get_name()
                    : "null");
}

}  // namespace sirius::op
