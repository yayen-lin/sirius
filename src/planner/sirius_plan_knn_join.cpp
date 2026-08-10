/*
 * Copyright 2025, Sirius Contributors.
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

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "helper/type_conversions.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/distance_metric.hpp"
#include "vss/vector_join_bind_data.hpp"

#include <cucascade/memory/common.hpp>

namespace sirius::planner {

namespace {

// A side is joinable only if its base table is resident on the GPU tier: the
// join reads its vectors and gathers its output columns straight from the
// pinned chunks. Refusing here (rather than at execute time) keeps an unpinned
// table a clean plan-time CPU fallback.
void require_pinned_on_gpu(duckdb::SiriusContext& sirius_ctx,
                           const sirius::vss::knn_join_side& side,
                           const char* which)
{
  const auto* pin = sirius_ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    side.catalog, side.schema, side.table);
  if (pin == nullptr || pin->tier != cucascade::memory::Tier::GPU) {
    throw duckdb::NotImplementedException(
      "sirius_knn_join: {} table '{}' is not pinned on the GPU tier", which, side.table);
  }
}

// TODO: build the GPU vector-join operator. The recognizer is wired up to this
// point; swapping this body for the real construction is what lands the operator.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> create_vector_join_operator(
  const sirius::vss::knn_join_bind_data& /*bind_data*/, duckdb::idx_t /*estimated_cardinality*/)
{
  throw duckdb::NotImplementedException(
    "sirius_knn_join: the GPU vector join operator is not implemented yet");
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan_knn_join(duckdb::LogicalGet& op)
{
  const auto* bind_data = dynamic_cast<const sirius::vss::knn_join_bind_data*>(op.bind_data.get());
  if (bind_data == nullptr) {
    throw duckdb::NotImplementedException("sirius_knn_join: missing or unexpected bind data");
  }
  const auto& req = bind_data->req;

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw duckdb::NotImplementedException(
      "sirius_knn_join requires the Sirius context to be initialized");
  }

  require_pinned_on_gpu(*sirius_ctx, req.probe, "probe");
  require_pinned_on_gpu(*sirius_ctx, req.corpus, "corpus");

  // search_mode => 'approx' searches a pinned cuVS index over the corpus vector
  // column; without one there is nothing to probe, so decline rather than
  // silently downgrading the user's request to a brute-force scan.
  if (req.use_index) {
    auto const metric = sirius::vss::ann_distance_type_from_metric(req.metric);
    if (sirius_ctx->get_cuvs_index_cache().find_by_column(
          req.corpus.table, req.corpus.column, metric) == nullptr) {
      throw duckdb::NotImplementedException(
        "sirius_knn_join: search_mode 'approx' needs an ANN index on '{}.{}' for metric '{}' — "
        "build one with sirius_create_ann_index",
        req.corpus.table,
        req.corpus.column,
        req.metric);
    }
  }

  op.ResolveOperatorTypes();

  // The operator emits every column the bind resolved, in bind order. The plan
  // may ask for a subset or a reordering, which a projection on top applies —
  // the same shape create_plan(LogicalGet&) uses for scans without projection
  // pushdown.
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> node =
    create_vector_join_operator(*bind_data, op.estimated_cardinality);

  auto const column_ids     = op.GetColumnIds();
  bool projection_necessary = column_ids.size() != bind_data->output_types.size();
  if (!projection_necessary) {
    for (std::size_t i = 0; i < column_ids.size(); ++i) {
      if (column_ids[i].GetPrimaryIndex() != i) {
        projection_necessary = true;
        break;
      }
    }
  }
  if (!projection_necessary) { return node; }

  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  types.reserve(column_ids.size());
  expressions.reserve(column_ids.size());
  for (auto const& column_id : column_ids) {
    if (!column_id.HasPrimaryIndex() || column_id.IsVirtualColumn() ||
        column_id.IsRowIdColumn() || column_id.IsEmptyColumn()) {
      throw duckdb::NotImplementedException(
        "sirius_knn_join: only plain output columns are supported");
    }
    auto const col_id = column_id.GetPrimaryIndex();
    if (col_id >= op.returned_types.size()) {
      throw duckdb::NotImplementedException("sirius_knn_join: column index out of range");
    }
    auto type = op.returned_types[col_id];
    types.push_back(type);
    expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, col_id));
  }

  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  select_list.reserve(expressions.size());
  for (auto& expr : expressions) {
    select_list.push_back(sirius::ast::from_duckdb(*expr));
  }
  return push_projection(std::move(node),
                         sirius::from_duckdb_vec(types),
                         std::move(select_list),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner
