#pragma once

#include "gpu_buffer_manager.hpp"
#include "gpu_physical_operator.hpp"
#include "graph/CachedCSR.hpp"
#include "graph/ParsedGraphQuery.hpp"

namespace duckdb {

// Forward declarations of CUDA kernel launchers
void LaunchEdgeTraversalKernel(const int64_t* offsets,
                               const int64_t* indices,
                               const int64_t* source_ids,
                               int64_t num_sources,
                               int64_t*& out_node_ids,
                               int64_t*& out_predecessor_ids,
                               int64_t& out_count);

void LaunchBFSKernel(const int64_t* csr_offsets,
                     const int64_t* csr_indices,
                     const int64_t* source_ids,
                     int64_t num_sources,
                     int64_t num_vertices,
                     int64_t*& out_node_ids,
                     int64_t*& out_distances,
                     int64_t*& out_predecessors,
                     int64_t& out_count);

class GPUGraphTraversalOperator : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

  GPUGraphTraversalOperator(vector<LogicalType> types,
                            const ParsedGraphQuery& parsed,
                            shared_ptr<CachedCSR> csr,
                            idx_t estimated_cardinality);

  // The parsed query - carries op type, source filter, output columns
  ParsedGraphQuery parsed;

  // CSR shared directly from GPUCSRConstructionOperator at construction time
  shared_ptr<CachedCSR> csr;

 public:
  // Source interface - runs traversal and writes results into output_relation
  SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;
  bool IsSource() const override { return true; }
  bool ParallelSource() const override { return false; }

 private:
  void RunEdgeTraversal(GPUIntermediateRelation& output_relation) const;
  void RunBFS(GPUIntermediateRelation& output_relation) const;

  // Result arrays populated by traversal, read by GetData
  mutable int64_t* result_node_ids        = nullptr;
  mutable int64_t* result_predecessor_ids = nullptr;
  mutable int64_t result_count            = 0;
  mutable bool traversal_done             = false;
  mutable int64_t* result_distances       = nullptr;
};

}  // namespace duckdb