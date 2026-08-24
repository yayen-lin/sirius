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

#include "op/sirius_physical_vector_threshold_join.hpp"

#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "vss/brute_force_threshold.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_resources.hpp>

#include <nvtx3/nvtx3.hpp>

#include <functional>
#include <vector>

namespace sirius {
namespace op {

sirius_physical_vector_threshold_join::sirius_physical_vector_threshold_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  std::size_t left_vector_col_idx,
  std::size_t right_vector_col_idx,
  float cutoff,
  std::string metric,
  std::int64_t dim,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::VECTOR_THRESHOLD_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    left_vector_col_idx(left_vector_col_idx),
    right_vector_col_idx(right_vector_col_idx),
    cutoff(cutoff),
    metric(std::move(metric)),
    dim(dim),
    join_type(join_type)
{
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto& lhs_types = children[0]->get_types();
  auto& rhs_types = children[1]->get_types();
  left_output_col_idxs.reserve(lhs_types.size());
  for (std::size_t i = 0; i < lhs_types.size(); i++) {
    left_output_col_idxs.push_back(i);
  }
  right_output_col_idxs.reserve(rhs_types.size());
  for (std::size_t i = 0; i < rhs_types.size(); i++) {
    right_output_col_idxs.push_back(i);
  }
}

void sirius_physical_vector_threshold_join::enable_distance_output(bool as_similarity)
{
  // Idempotent: the planner may match more than one distance reference in the same SELECT list.
  if (emit_distance_) { return; }
  emit_distance_               = true;
  emit_distance_as_similarity_ = as_similarity;
  // The join carries no physical-type overrides, so the logical types fully describe the output
  // batches. Append one FLOAT column for the distance; it lands at output index n_left + n_right,
  // which is what the planner rewrites its distance reference to.
  D_ASSERT(!has_physical_overrides());
  types.push_back(sirius::from_duckdb(duckdb::LogicalType::FLOAT));
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_vector_threshold_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Mirrors sirius_physical_nested_loop_join::build_pipelines.
  pipeline::sirius_meta_pipeline* host_meta;
  pipeline::sirius_pipeline* host_current;
  if (is_sink()) {
    auto& sink_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
    host_meta       = &sink_meta;
    host_current    = sink_meta.get_base_pipeline().get();
  } else {
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    host_meta    = &meta_pipeline;
    host_current = &current;
  }

  D_ASSERT(children.size() == 2);
  auto& build_child = *children[1];
  D_ASSERT(build_child.is_sink());
  D_ASSERT(!build_child.children.empty());
  auto& build_meta = host_meta->create_child_meta_pipeline(*host_current, build_child);
  build_meta.build(*build_child.children[0]);

  auto& probe_child = *children[0];
  D_ASSERT(probe_child.is_sink());
  D_ASSERT(!probe_child.children.empty());
  auto& probe_meta = host_meta->create_child_meta_pipeline(*host_current, probe_child);
  probe_meta.build(*probe_child.children[0]);
}

partition_strategy sirius_physical_vector_threshold_join::get_partition_strategy(
  const partition_sizing_input& /*in*/)
{
  // Streams both sides through per-batch-pair threshold joins on a single partition; the answer is
  // the union of the per-pair edge lists, so it never hash-partitions, broadcasts, or build-probes.
  return {/*num_partitions=*/1, /*broadcast=*/false, /*build_probe=*/false};
}

std::unique_ptr<operator_data> sirius_physical_vector_threshold_join::get_next_task_input_data()
{
  // Mirrors sirius_physical_nested_loop_join::get_next_task_input_data: enumerate every
  // (left batch, right batch) pair, popping the last consumer of each batch.
  std::lock_guard<std::mutex> lg(batches_to_processed_mutex);

  if (left_batch_ids.empty() && right_batch_ids.empty()) {
    auto* default_port = get_port("default");
    auto* build_port   = get_port("build");
    if (!default_port || !default_port->repo || !build_port || !build_port->repo) {
      return nullptr;
    }
    if (default_port->repo->num_partitions() != build_port->repo->num_partitions()) {
      throw std::runtime_error(
        "sirius_physical_vector_threshold_join: number of partitions for default and build ports "
        "must match");
    }
    left_batch_ids.reserve(default_port->repo->num_partitions());
    right_batch_ids.reserve(build_port->repo->num_partitions());
    for (size_t i = 0; i < default_port->repo->num_partitions(); i++) {
      left_batch_ids.push_back(default_port->repo->get_batch_ids(i));
      right_batch_ids.push_back(build_port->repo->get_batch_ids(i));
      num_batches_to_process += left_batch_ids[i].size() * right_batch_ids[i].size();
    }
  }

  if (current_partition_index >= num_batches_to_process) { return nullptr; }

  size_t batch_index = current_partition_index++;

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  size_t counter     = 0;
  auto* default_port = get_port("default");
  auto* build_port   = get_port("build");
  for (size_t partition_idx = 0; partition_idx < left_batch_ids.size(); partition_idx++) {
    size_t left_counter = 0;
    for (auto& left_batch_id : left_batch_ids[partition_idx]) {
      size_t right_counter = 0;
      for (auto& right_batch_id : right_batch_ids[partition_idx]) {
        if (counter == batch_index) {
          if (right_counter == right_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(
              default_port->repo->pop_data_batch_by_id(left_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              default_port->repo->get_data_batch_by_id(left_batch_id, partition_idx));
          }
          if (left_counter == left_batch_ids[partition_idx].size() - 1) {
            input_batch.push_back(
              build_port->repo->pop_data_batch_by_id(right_batch_id, partition_idx));
          } else {
            input_batch.push_back(
              build_port->repo->get_data_batch_by_id(right_batch_id, partition_idx));
          }
          return std::make_unique<pipelineable_operator_data>(input_batch);
        }
        right_counter++;
        counter++;
      }
      left_counter++;
    }
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_vector_threshold_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_threshold_join::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() != 2) {
    throw std::runtime_error(
      "sirius_physical_vector_threshold_join expects 2 input batches (left, right), got " +
      std::to_string(input_batches.size()));
  }

  auto const& left_batch  = input_batches[0];
  auto const& right_batch = input_batches[1];

  cudf::table_view left  = get_cudf_table_view(left_batch);
  cudf::table_view right = get_cudf_table_view(right_batch);

  cucascade::memory::memory_space* space = left_batch.get_memory_space();
  if (!space) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  auto mr = space->get_default_allocator();

  bool const is_left     = join_type == duckdb::JoinType::LEFT;
  auto const n_left      = static_cast<cudf::size_type>(left.num_rows());
  bool const right_empty = right.num_rows() == 0;

  // No left rows -> no output. An empty right side yields no matches: INNER emits nothing, while
  // LEFT still emits every left row padded with NULLs (handled by the unmatched pass below).
  if (n_left == 0 || (right_empty && !is_left)) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // Assemble one output table from a left gather-map and an equal-length right gather-map, taking
  // the plan's output columns in [left..., right...] order (mirrors nested_loop_join output).
  // When emit_distance_ is set the caller passes a same-length FLOAT column of per-pair distances,
  // appended after the right columns at output index n_left + n_right. The unmatched (LEFT) pass
  // passes an all-NULL distance column so the two parts share one schema for the concatenate below.
  auto assemble = [&](cudf::column_view const& left_map,
                      cudf::column_view const& right_map,
                      cudf::out_of_bounds_policy right_policy,
                      std::unique_ptr<cudf::column> distance_col) {
    auto left_gathered =
      cudf::gather(left, left_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto right_gathered = cudf::gather(right, right_map, right_policy, stream, mr);
    auto left_released  = left_gathered->release();
    auto right_released = right_gathered->release();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size() +
                 (emit_distance_ ? 1 : 0));
    for (std::size_t idx : left_output_col_idxs) {
      if (idx < left_released.size()) { cols.push_back(std::move(left_released[idx])); }
    }
    for (std::size_t idx : right_output_col_idxs) {
      if (idx < right_released.size()) { cols.push_back(std::move(right_released[idx])); }
    }
    if (emit_distance_) { cols.push_back(std::move(distance_col)); }
    return std::make_unique<cudf::table>(std::move(cols));
  };

  std::vector<std::unique_ptr<cudf::table>> output_parts;

  // Matched pairs: the tiled-GEMM threshold kernel emits an edge list (query_rows = local left row,
  // neighbors = local right row). Skipped when the right side is empty (no matches possible).
  std::unique_ptr<cudf::column> matched_left_map;
  if (!right_empty) {
    auto const queries = vss::list_column_as_dataset_view(left.column(left_vector_col_idx), dim);
    auto const dataset = vss::list_column_as_dataset_view(right.column(right_vector_col_idx), dim);
    raft::device_resources res{stream};
    auto const metric_type =
      vss::join_selection_distance_type_from_metric(metric, /*exact_unexpanded=*/false);
    auto tj = vss::brute_force_threshold(res, dataset, queries, cutoff, metric_type, mr);
    // Reuse the distances the kernel already computed as the output column. array_cosine_similarity
    // wants `1 - cosine_distance`; array_distance / array_cosine_distance want the value as-is.
    std::unique_ptr<cudf::column> distance_col;
    if (emit_distance_) {
      if (emit_distance_as_similarity_) {
        cudf::numeric_scalar<float> one(1.0F, true, stream);
        distance_col = cudf::binary_operation(one, tj.distances->view(),
                                              cudf::binary_operator::SUB,
                                              cudf::data_type{cudf::type_id::FLOAT32}, stream, mr);
      } else {
        distance_col = std::move(tj.distances);
      }
    }
    output_parts.push_back(assemble(tj.query_rows->view(), tj.neighbors->view(),
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    std::move(distance_col)));
    matched_left_map = std::move(tj.query_rows);
  }

  // LEFT: append every left row with no match, padded with NULL right columns. (For LEFT the right
  // side is folded to one batch, so "no match here" means no match at all.)
  if (is_left) {
    // Per-left-row matched flag: true where the row produced at least one edge.
    cudf::numeric_scalar<bool> false_scalar(false, true, stream);
    cudf::numeric_scalar<bool> true_scalar(true, true, stream);
    auto matched_flag = cudf::make_column_from_scalar(false_scalar, n_left, stream, mr);
    if (matched_left_map && matched_left_map->size() > 0) {
      auto scattered =
        cudf::scatter({std::ref(static_cast<cudf::scalar const&>(true_scalar))},
                      matched_left_map->view(), cudf::table_view({matched_flag->view()}), stream);
      matched_flag = std::move(scattered->release()[0]);
    }
    auto unmatched_mask =
      cudf::unary_operation(matched_flag->view(), cudf::unary_operator::NOT, stream, mr);
    // Local left row indices that survived (unmatched), via sequence[0,n_left) filtered by the mask.
    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
    cudf::numeric_scalar<cudf::size_type> one(1, true, stream);
    auto seq               = cudf::sequence(n_left, zero, one, stream, mr);
    auto unmatched_idx_tbl = cudf::apply_boolean_mask(
      cudf::table_view({seq->view()}), unmatched_mask->view(), stream, mr);
    auto unmatched_idx = unmatched_idx_tbl->get_column(0).view();
    // All-NULL right columns: gather every right row by an out-of-range index under NULLIFY. Use
    // the upper bound (num_rows), not -1: this cudf's NULLIFY only nullifies indices >= num_rows,
    // and treats a negative index as an in-range wrap (returning the last row instead of NULL).
    cudf::numeric_scalar<cudf::size_type> oob(static_cast<cudf::size_type>(right.num_rows()), true,
                                              stream);
    auto pad = cudf::make_column_from_scalar(
      oob, static_cast<cudf::size_type>(unmatched_idx.size()), stream, mr);
    // Unmatched left rows have no partner, so their distance is NULL.
    std::unique_ptr<cudf::column> distance_col;
    if (emit_distance_) {
      distance_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                               static_cast<cudf::size_type>(unmatched_idx.size()),
                                               cudf::mask_state::ALL_NULL, stream, mr);
    }
    output_parts.push_back(assemble(unmatched_idx, pad->view(),
                                    cudf::out_of_bounds_policy::NULLIFY, std::move(distance_col)));
  }

  std::unique_ptr<cudf::table> result_table;
  if (output_parts.size() == 1) {
    result_table = std::move(output_parts[0]);
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(output_parts.size());
    for (auto& part : output_parts) { views.push_back(part->view()); }
    result_table = cudf::concatenate(views, stream, mr);
  }

  auto batch = make_data_batch(std::move(result_table), *space, stream, batch_telemetry());
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
}

}  // namespace op
}  // namespace sirius
