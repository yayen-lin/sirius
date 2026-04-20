#pragma once

#include "graph/CachedCSR.hpp"
#include "graph/ParsedGraphQuery.hpp"
#include "op/sirius_physical_operator.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <duckdb.hpp>

/* forward declarations for CUDA kernels (src/cuda/operator/) */
namespace duckdb {

int64_t LaunchFindMaxKernel(const int64_t* data, int64_t n);

void LaunchDegreeCountKernel(const int64_t* src_col,
                             int64_t* degree,
                             int64_t num_edges,
                             int64_t num_vertices);

void LaunchPrefixScanKernel(const int64_t* degree, int64_t* offsets, int64_t num_vertices);

void LaunchScatterKernel(const int64_t* src_col,
                         const int64_t* dst_col,
                         const int64_t* offsets,
                         int64_t* indices,
                         int64_t num_edges,
                         int64_t num_vertices);

}  // namespace duckdb

namespace sirius::op {

// Build the CSR from pending_batches inside csr if not yet built.
// Safe to call from within operator::execute() where a live reservation is held.
void build_csr_if_needed(duckdb::shared_ptr<duckdb::CachedCSR>& csr, rmm::cuda_stream_view stream);

class GPUCSRConstructionOperator : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::CSR_CONSTRUCTION;

  GPUCSRConstructionOperator(duckdb::vector<duckdb::LogicalType> types,
                             const duckdb::ParsedGraphQuery& parsed,
                             duckdb::idx_t estimated_cardinality,
                             cucascade::memory::memory_space& gpu_memory_space);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void finalize_operator() override;

  bool is_sink() const override { return true; }

  std::string params_to_string() const override;

  duckdb::ParsedGraphQuery parsed;

  // shared with GPUGraphTraversalOperator; traversal calls build_csr_if_needed() on it
  mutable duckdb::shared_ptr<duckdb::CachedCSR> csr;

 private:
  cucascade::memory::memory_space* _gpu_memory_space = nullptr;
};

}  // namespace sirius::op
