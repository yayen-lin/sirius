#pragma once

#include "graph/CachedCSR.hpp"
#include "graph/ParsedGraphQuery.hpp"
#include "op/sirius_physical_operator.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <duckdb.hpp>

/* forward declarations (src/cuda/operator/) */
namespace duckdb {

// parallel degree counting with atomics
void LaunchDegreeCountKernel(const int64_t* src_col,
                             int64_t* degree,
                             int64_t num_edges,
                             int64_t num_vertices);

// HSO two-pass exclusive scan
void LaunchPrefixScanKernel(const int64_t* degree, int64_t* offsets, int64_t num_vertices);

// parallel edge scatter into CSR indices
void LaunchScatterKernel(const int64_t* src_col,
                         const int64_t* dst_col,
                         const int64_t* offsets,
                         int64_t* indices,
                         int64_t num_edges,
                         int64_t num_vertices);
}  // namespace duckdb

namespace sirius::op {

class GPUCSRConstructionOperator : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::CSR_CONSTRUCTION;

  GPUCSRConstructionOperator(duckdb::vector<duckdb::LogicalType> types,
                             const duckdb::ParsedGraphQuery& parsed,
                             duckdb::idx_t estimated_cardinality);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  bool is_sink() const override { return true; }

  std::string params_to_string() const override;

  // parsed query (includes edge_table, src_col, dst_col)
  duckdb::ParsedGraphQuery parsed;

  // constructed (or cache-hit) CSR, set during sink/execute
  mutable duckdb::shared_ptr<duckdb::CachedCSR> csr;

 private:
  void BuildCSR(const std::vector<std::shared_ptr<::cucascade::data_batch>>& edge_batches,
                rmm::cuda_stream_view stream) const;
};

}  // namespace sirius::op
