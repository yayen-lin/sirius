#pragma once

#include "graph/sirius_cached_csr.hpp"
#include "graph/sirius_parsed_graph_query.hpp"
#include "op/sirius_physical_operator.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <duckdb.hpp>

/* forward declarations for CUDA kernels (src/cuda/operator/) */
namespace duckdb {

int64_t LaunchFindMaxKernel(const int64_t* data, int64_t n);

void LaunchEdgeSortKernel(const int64_t* src_in,
                         const int64_t* dst_in,
                         int64_t* src_out,
                         int64_t* dst_out,
                         int64_t num_edges,
                         rmm::cuda_stream_view stream);

void LaunchDegreeCountKernel(const int64_t* src_col,
                             int64_t* degree,
                             int64_t num_edges,
                             int64_t num_vertices);

void LaunchPrefixScanKernel(const int64_t* degree,
                            int64_t* offsets,
                            int64_t num_vertices,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

void LaunchScatterKernel(const int64_t* src_col,
                         const int64_t* dst_col,
                         const int64_t* offsets,
                         int64_t* indices,
                         int64_t num_edges,
                         int64_t num_vertices,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);

void LaunchFillKernel(int64_t* data, int64_t value, size_t n, rmm::cuda_stream_view stream);

}  // namespace duckdb

namespace sirius::op {

// Build the CSR from pending_batches inside csr if not yet built
void build_csr_if_needed(duckdb::shared_ptr<duckdb::sirius_cached_csr>& csr, rmm::cuda_stream_view stream);

class sirius_physical_csr_construction : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::CSR_CONSTRUCTION;

  sirius_physical_csr_construction(duckdb::vector<duckdb::LogicalType> types,
                                   const duckdb::sirius_parsed_graph_query& parsed,
                                   duckdb::idx_t estimated_cardinality,
                                   cucascade::memory::memory_space& gpu_memory_space);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void finalize_operator() override;

  bool is_sink() const override { return true; }

  std::string params_to_string() const override;

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  duckdb::sirius_parsed_graph_query parsed;

  // shared with sirius_physical_graph_traversal; traversal calls build_csr_if_needed() on it
  mutable duckdb::shared_ptr<duckdb::sirius_cached_csr> csr;

 private:
  cucascade::memory::memory_space* _gpu_memory_space = nullptr;
};

}  // namespace sirius::op
