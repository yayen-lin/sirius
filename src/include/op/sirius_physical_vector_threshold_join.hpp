/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

//! sirius_physical_vector_threshold_join is the SQL-integrated threshold (radius) vector join.
//! It recognizes `l JOIN r ON array_distance(l.v, r.v) <= eps` and runs it on the GPU by consuming
//! the two table scans as children (no pinning) and applying the tiled-GEMM threshold kernel per
//! (left batch, right batch) pair. It mirrors sirius_physical_nested_loop_join's build/probe shape;
//! the per-pair core is `vss::brute_force_threshold` instead of `cudf::conditional_join`. Threshold
//! edges are independent across batch pairs, so the union of the per-pair outputs is the answer --
//! no cross-batch merge.
class sirius_physical_vector_threshold_join : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_THRESHOLD_JOIN;

 public:
  sirius_physical_vector_threshold_join(duckdb::LogicalOperator& op,
                                        duckdb::unique_ptr<sirius_physical_operator> left,
                                        duckdb::unique_ptr<sirius_physical_operator> right,
                                        std::size_t left_vector_col_idx,
                                        std::size_t right_vector_col_idx,
                                        float cutoff,
                                        std::string metric,
                                        std::int64_t dim,
                                        duckdb::JoinType join_type,
                                        std::size_t estimated_cardinality);

  //! Column index of the FLOAT[dim] vector column within the left (probe) child's output.
  std::size_t left_vector_col_idx;
  //! Column index of the FLOAT[dim] vector column within the right (build) child's output.
  std::size_t right_vector_col_idx;
  //! Distance cutoff in the metric's own units (cosine similarity is pre-flipped to 1 - eps).
  float cutoff;
  //! Distance metric: "l2" or "cosine".
  std::string metric;
  //! Fixed vector dimensionality (from the ARRAY<FLOAT> logical type).
  std::int64_t dim;
  //! The join type (INNER for milestone 1).
  duckdb::JoinType join_type;

  //! Output column order: identity over the left child's columns.
  duckdb::vector<std::size_t> left_output_col_idxs;
  //! Output column order: identity over the right child's columns.
  duckdb::vector<std::size_t> right_output_col_idxs;

  //! Ask the join to emit the per-pair distance as one extra trailing FLOAT column. The planner
  //! calls this when the SELECT list references the same distance function as the join predicate,
  //! so the value can be reused instead of recomputed in a projection. When @p as_similarity is
  //! true the emitted value is `1 - distance` (matching array_cosine_similarity); otherwise it is
  //! the raw metric distance (array_distance / array_cosine_distance). Idempotent. Appends one
  //! FLOAT to the output types, so the distance lands at output index n_left + n_right.
  void enable_distance_output(bool as_similarity);

 protected:
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

 public:
  //! Always a source: every join emits output.
  bool is_source() const override { return true; }

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! Streams both sides through per-batch-pair threshold joins on a single partition; never
  //! hash-partitions, broadcasts, or enters build-probe.
  partition_strategy get_partition_strategy(const partition_sizing_input& in) override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 protected:
  std::mutex batches_to_processed_mutex;
  std::size_t current_partition_index = 0;
  std::size_t num_batches_to_process  = 0;
  std::vector<std::vector<uint64_t>> left_batch_ids;
  std::vector<std::vector<uint64_t>> right_batch_ids;

  //! Set by enable_distance_output(): emit the pairwise distance as a trailing FLOAT column.
  bool emit_distance_ = false;
  //! When emitting the distance, output `1 - distance` instead of the raw metric distance.
  bool emit_distance_as_similarity_ = false;
};

}  // namespace op
}  // namespace sirius
