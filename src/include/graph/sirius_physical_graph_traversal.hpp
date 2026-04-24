#pragma once

#include "graph/sirius_cached_csr.hpp"
#include "graph/sirius_parsed_graph_query.hpp"
#include "op/sirius_physical_operator.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <duckdb.hpp>

/* forward declarations (src/cuda/operator/) */
namespace duckdb {  // TODO: add more operators like multi-source bfs, SSSP, etc

// result structures for graph kernels (RAII-safe, integrates with Sirius memory management)
struct EdgeTraversalResult {
  rmm::device_uvector<int64_t> node_ids;
  rmm::device_uvector<int64_t> predecessor_ids;

  EdgeTraversalResult(rmm::device_uvector<int64_t>&& nodes,
                     rmm::device_uvector<int64_t>&& predecessors);
};

struct BFSResult {
  rmm::device_uvector<int64_t> node_ids;
  rmm::device_uvector<int64_t> distances;
  rmm::device_uvector<int64_t> predecessors;

  BFSResult(rmm::device_uvector<int64_t>&& nodes,
           rmm::device_uvector<int64_t>&& dists,
           rmm::device_uvector<int64_t>&& preds);
};

// graph kernel launchers
EdgeTraversalResult LaunchEdgeTraversalKernel(const int64_t* offsets,
                                               const int64_t* indices,
                                               const int64_t* source_ids,
                                               int64_t num_sources,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

BFSResult LaunchBFSKernel(const int64_t* csr_offsets,
                          const int64_t* csr_indices,
                          const int64_t* source_ids,
                          int64_t num_sources,
                          int64_t num_vertices,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr);
}  // namespace duckdb

namespace sirius::op {

class sirius_physical_graph_traversal : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::GRAPH_TRAVERSAL;

  sirius_physical_graph_traversal(duckdb::vector<duckdb::LogicalType> types,
                                  const duckdb::sirius_parsed_graph_query& parsed,
                                  duckdb::shared_ptr<duckdb::sirius_cached_csr> csr,
                                  duckdb::idx_t estimated_cardinality,
                                  cucascade::memory::memory_space& gpu_memory_space);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  std::string params_to_string() const override;

  // parsed query (includes op type, source filter, output columns)
  duckdb::sirius_parsed_graph_query parsed;

  // CSR shared from sirius_physical_csr_construction
  duckdb::shared_ptr<duckdb::sirius_cached_csr> csr;

 private:
  std::unique_ptr<operator_data> RunEdgeTraversal(rmm::cuda_stream_view stream) const;
  std::unique_ptr<operator_data> RunBFS(rmm::cuda_stream_view stream) const;

  // GPU memory for the constructed CSR
  mutable cucascade::memory::memory_space* _gpu_memory_space = nullptr;

  // track if traversal has been executed
  mutable bool traversal_done = false;

  // cached results from first execute() call
  mutable std::unique_ptr<operator_data> cached_result;
};

}  // namespace sirius::op
