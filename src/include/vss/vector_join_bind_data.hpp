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

#pragma once

#include <helper/logical_type.hpp>
#include <vss/vector_join.hpp>

#include <duckdb/function/table_function.hpp>

namespace sirius::vss {

/// Bind data of a sirius_knn_join call. The TVF body never runs on the GPU
/// path: the plan generator recognizes the LogicalGet, casts its bind data to
/// this, and builds the vector join operator from the request.
struct knn_join_bind_data : public duckdb::TableFunctionData {
  knn_join_request req;
  /// Output column types: probe columns, corpus columns, trailing distance.
  duckdb::vector<sirius::logical_type> output_types;
};

}  // namespace sirius::vss
