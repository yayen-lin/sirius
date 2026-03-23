#pragma once

#include "gpu_buffer_manager.hpp"
#include "gpu_physical_operator.hpp"
#include "graph/CachedCSR.hpp"
#include "graph/ParsedGraphQuery.hpp"

namespace duckdb {

// Forward declarations of CUDA kernel launchers (defined in src/cuda/operator/)
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

class GPUCSRConstructionOperator : public GPUPhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

  GPUCSRConstructionOperator(vector<LogicalType> types,
                             const ParsedGraphQuery& parsed,
                             idx_t estimated_cardinality);

  // The parsed query - carries edge_table, src_col, dst_col
  ParsedGraphQuery parsed;

  // The constructed (or cache-hit) CSR, set during Sink/CombineFinalize
  mutable shared_ptr<CachedCSR> csr;

public:
  // Sink interface - receives edge columns from GPUPhysicalTableScan
  SinkResultType Sink(GPUIntermediateRelation& input_relation) const override;
  SinkFinalizeType CombineFinalize(vector<shared_ptr<GPUIntermediateRelation>>& input,
                                   GPUIntermediateRelation& output) const override;
  bool IsSink() const override { return true; }
  bool ParallelSink() const override { return false; }

  // Source interface - passes CachedCSR pointer downstream to GPUGraphTraversalOperator
  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;
  bool IsSource() const override { return true; }
  bool ParallelSource() const override { return false; }

private:
  void BuildCSR(GPUIntermediateRelation& edge_relation) const;
};

}  // namespace duckdb